#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile uint64_t probe_sink;

static double now_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    perror("clock_gettime");
    exit(2);
  }
  return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;
}

static void *aligned_allocate(size_t bytes) {
  void *pointer = NULL;
  int error = posix_memalign(&pointer, 64, bytes);
  if (error != 0 || pointer == NULL) {
    fprintf(stderr, "allocation failed for %zu bytes: %s\n", bytes,
            strerror(error == 0 ? errno : error));
    exit(2);
  }
  return pointer;
}

static uint64_t xorshift64(uint64_t *state) {
  uint64_t value = *state;
  value ^= value << 13;
  value ^= value >> 7;
  value ^= value << 17;
  *state = value;
  return value;
}

static void initialize_bytes(uint8_t *data, size_t bytes) {
  for (size_t offset = 0; offset < bytes; offset += 64)
    data[offset] = (uint8_t)(offset >> 6);
}

static double measure_pointer_chase(size_t bytes, uint64_t iterations) {
  size_t count = bytes / sizeof(uint32_t);
  if (count < 2)
    return 0.0;

  uint32_t *next = aligned_allocate(count * sizeof(uint32_t));
  uint32_t *order = aligned_allocate(count * sizeof(uint32_t));
  for (size_t index = 0; index < count; ++index)
    order[index] = (uint32_t)index;

  uint64_t random_state = UINT64_C(0x9e3779b97f4a7c15) ^ bytes;
  for (size_t index = count - 1; index > 0; --index) {
    size_t other = (size_t)(xorshift64(&random_state) % (index + 1));
    uint32_t temporary = order[index];
    order[index] = order[other];
    order[other] = temporary;
  }
  for (size_t index = 0; index < count; ++index)
    next[order[index]] = order[(index + 1) % count];

  uint32_t cursor = order[0];
  uint64_t warmup = count < UINT64_C(1000000) ? count : UINT64_C(1000000);
  for (uint64_t index = 0; index < warmup; ++index)
    cursor = next[cursor];

  double started = now_ns();
  for (uint64_t index = 0; index < iterations; ++index)
    cursor = next[cursor];
  double elapsed = now_ns() - started;
  probe_sink += cursor;

  free(order);
  free(next);
  return elapsed / (double)iterations;
}

static double measure_bandwidth(size_t bytes, unsigned rounds) {
  uint64_t *data = aligned_allocate(bytes);
  size_t count = bytes / sizeof(uint64_t);
  for (size_t index = 0; index < count; ++index)
    data[index] = index * UINT64_C(0x9e3779b97f4a7c15);

  uint64_t sum = 0;
  for (size_t index = 0; index < count; ++index)
    sum += data[index];

  double started = now_ns();
  for (unsigned round = 0; round < rounds; ++round) {
    for (size_t index = 0; index < count; ++index)
      sum += data[index];
  }
  double elapsed = now_ns() - started;
  probe_sink += sum;
  free(data);
  return ((double)bytes * (double)rounds) / elapsed;
}

static double measure_stride(size_t bytes, size_t stride) {
  uint8_t *data = aligned_allocate(bytes);
  initialize_bytes(data, bytes);
  uint64_t accesses = bytes / stride;
  if (accesses == 0)
    accesses = 1;
  uint64_t sum = 0;

  double started = now_ns();
  for (size_t offset = 0; offset < bytes; offset += stride)
    sum += data[offset];
  double elapsed = now_ns() - started;
  probe_sink += sum;
  free(data);
  return elapsed / (double)accesses;
}

__attribute__((noinline)) static double
measure_prefetch_loop(size_t bytes, uint64_t iterations, unsigned issue_every,
                      int enable_prefetch) {
  uint8_t *data = aligned_allocate(bytes);
  initialize_bytes(data, bytes);
  size_t mask = bytes - 1;
  size_t offset = 0;
  uint64_t sum = 0;

  double started = now_ns();
  for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
    offset = (offset + 64) & mask;
    if (enable_prefetch && iteration % issue_every == 0)
      __builtin_prefetch(data + ((offset + 8 * 64) & mask), 0, 3);
    sum += data[offset];
  }
  double elapsed = now_ns() - started;
  probe_sink += sum;
  free(data);
  return elapsed / (double)iterations;
}

static void run_latency(int quick) {
  const size_t sizes[] = {4 * 1024,       16 * 1024,      32 * 1024,
                          64 * 1024,      256 * 1024,     1024 * 1024,
                          4 * 1024 * 1024, 16 * 1024 * 1024,
                          64 * 1024 * 1024};
  const size_t size_count = sizeof(sizes) / sizeof(sizes[0]);
  uint64_t iterations = quick ? UINT64_C(400000) : UINT64_C(2000000);
  for (size_t index = 0; index < size_count; ++index) {
    double ns_per_load = measure_pointer_chase(sizes[index], iterations);
    printf("latency\t%zu\t%.9f\n", sizes[index], ns_per_load);
  }
}

static void run_bandwidth(int quick) {
  size_t bytes = quick ? 64 * 1024 * 1024 : 256 * 1024 * 1024;
  unsigned rounds = quick ? 3 : 6;
  double bytes_per_ns = measure_bandwidth(bytes, rounds);
  printf("bandwidth\t%zu\t%u\t%.9f\n", bytes, rounds, bytes_per_ns);
}

static void run_stride(int quick) {
  const size_t strides[] = {64, 128, 256, 512, 1024, 2048, 4096};
  const size_t stride_count = sizeof(strides) / sizeof(strides[0]);
  size_t bytes = quick ? 32 * 1024 * 1024 : 128 * 1024 * 1024;
  for (size_t index = 0; index < stride_count; ++index) {
    double ns_per_access = measure_stride(bytes, strides[index]);
    printf("stride\t%zu\t%.9f\n", strides[index], ns_per_access);
  }
}

static void run_prefetch_cost(int quick) {
  size_t bytes = 32 * 1024;
  uint64_t iterations = quick ? UINT64_C(2000000) : UINT64_C(10000000);
  double baseline = measure_prefetch_loop(bytes, iterations, 1, 0);
  printf("prefetch\t0\t%.9f\n", baseline);
  const unsigned frequencies[] = {1, 2, 4};
  for (size_t index = 0;
       index < sizeof(frequencies) / sizeof(frequencies[0]); ++index) {
    unsigned issue_every = frequencies[index];
    double measured =
        measure_prefetch_loop(bytes, iterations, issue_every, 1);
    printf("prefetch\t%u\t%.9f\n", issue_every, measured);
  }
}

int main(int argc, char **argv) {
  int quick = 0;
  if (argc == 2 && strcmp(argv[1], "--quick") == 0)
    quick = 1;
  else if (argc != 1) {
    fprintf(stderr, "usage: %s [--quick]\n", argv[0]);
    return 2;
  }

  run_latency(quick);
  run_bandwidth(quick);
  run_stride(quick);
  run_prefetch_cost(quick);
  fprintf(stderr, "probe_sink=%" PRIu64 "\n", probe_sink);
  return 0;
}
