#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void sme_measure_loop_baseline(uint64_t groups, uint64_t *ticks,
                                      uint64_t *frequency);
extern void sme_measure_fmopa_one_tile(uint64_t groups, uint64_t *ticks,
                                      uint64_t *frequency);
extern void sme_measure_fmopa_four_tiles(uint64_t groups, uint64_t *ticks,
                                        uint64_t *frequency);
extern uint64_t sme_streaming_vector_bytes(void);
extern void sme_run_loop_baseline(uint64_t groups);
extern void sme_run_fmopa_one_tile(uint64_t groups);
extern void sme_run_fmopa_four_tiles(uint64_t groups);

typedef void (*probe_function)(uint64_t, uint64_t *, uint64_t *);
typedef void (*clock_probe_function)(uint64_t);

static uint64_t now_ns(void) {
  struct timespec timestamp;
#if defined(CLOCK_MONOTONIC_RAW)
  const clockid_t clock_id = CLOCK_MONOTONIC_RAW;
#else
  const clockid_t clock_id = CLOCK_MONOTONIC;
#endif
  if (clock_gettime(clock_id, &timestamp) != 0) {
    perror("clock_gettime");
    exit(2);
  }
  return (uint64_t)timestamp.tv_sec * UINT64_C(1000000000) +
         (uint64_t)timestamp.tv_nsec;
}

static void run_probe(const char *name, probe_function function,
                      uint64_t groups, unsigned operations_per_group,
                      unsigned repetitions) {
  for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
    uint64_t ticks = 0;
    uint64_t frequency = 0;
    function(groups, &ticks, &frequency);
    if (ticks == 0 || frequency == 0) {
      fprintf(stderr, "architected timer returned zero\n");
      exit(2);
    }
    printf("timing\t%s\t%u\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
           "\n",
           name, operations_per_group, groups, ticks, frequency);
  }
}

static void run_clock_probe(const char *name, clock_probe_function function,
                            uint64_t groups, unsigned operations_per_group,
                            unsigned repetitions) {
  for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
    uint64_t started = now_ns();
    function(groups);
    uint64_t elapsed = now_ns() - started;
    printf("timing\t%s\t%u\t%" PRIu64 "\t%" PRIu64
           "\t1000000000\n",
           name, operations_per_group, groups, elapsed);
  }
}

int main(int argc, char **argv) {
  uint64_t groups = UINT64_C(1000000);
  unsigned repetitions = 7;
  int use_clock = 0;
  if (argc >= 2) {
    char *end = NULL;
    groups = strtoull(argv[1], &end, 10);
    if (!end || *end != '\0' || groups == 0) {
      fprintf(stderr, "groups must be a positive integer\n");
      return 2;
    }
  }
  if (argc == 3 && strcmp(argv[2], "--clock") == 0)
    use_clock = 1;
  else if (argc > 2) {
    fprintf(stderr, "usage: %s [groups] [--clock]\n", argv[0]);
    return 2;
  }

  printf("system\tstreaming_vector_bytes\t%" PRIu64 "\n",
         sme_streaming_vector_bytes());
  if (use_clock) {
    sme_run_fmopa_four_tiles(10000);
    printf("system\ttimer\tCLOCK_MONOTONIC_RAW\n");
    run_clock_probe("baseline", sme_run_loop_baseline, groups, 0, repetitions);
    run_clock_probe("fmopa_one_tile", sme_run_fmopa_one_tile, groups, 1,
                    repetitions);
    run_clock_probe("fmopa_four_tiles", sme_run_fmopa_four_tiles, groups, 4,
                    repetitions);
  } else {
    uint64_t ignored_ticks = 0;
    uint64_t ignored_frequency = 0;
    sme_measure_fmopa_four_tiles(10000, &ignored_ticks, &ignored_frequency);
    printf("system\ttimer\tCNTVCT_EL0\n");
    run_probe("baseline", sme_measure_loop_baseline, groups, 0, repetitions);
    run_probe("fmopa_one_tile", sme_measure_fmopa_one_tile, groups, 1,
              repetitions);
    run_probe("fmopa_four_tiles", sme_measure_fmopa_four_tiles, groups, 4,
              repetitions);
  }
  return 0;
}
