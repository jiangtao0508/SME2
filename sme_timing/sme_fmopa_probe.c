#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void sme_measure_loop_baseline(uint64_t groups, uint64_t *ticks,
                                      uint64_t *frequency);
extern void sme_measure_fmopa_one_tile(uint64_t groups, uint64_t *ticks,
                                      uint64_t *frequency);
extern void sme_measure_fmopa_four_tiles(uint64_t groups, uint64_t *ticks,
                                        uint64_t *frequency);
extern uint64_t sme_streaming_vector_bytes(void);

typedef void (*probe_function)(uint64_t, uint64_t *, uint64_t *);

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

int main(int argc, char **argv) {
  uint64_t groups = UINT64_C(1000000);
  unsigned repetitions = 7;
  if (argc == 2) {
    char *end = NULL;
    groups = strtoull(argv[1], &end, 10);
    if (!end || *end != '\0' || groups == 0) {
      fprintf(stderr, "groups must be a positive integer\n");
      return 2;
    }
  } else if (argc != 1) {
    fprintf(stderr, "usage: %s [groups]\n", argv[0]);
    return 2;
  }

  // Warm up streaming mode, ZA allocation and the instruction path.
  uint64_t ignored_ticks = 0;
  uint64_t ignored_frequency = 0;
  sme_measure_fmopa_four_tiles(10000, &ignored_ticks, &ignored_frequency);

  printf("system\tstreaming_vector_bytes\t%" PRIu64 "\n",
         sme_streaming_vector_bytes());

  run_probe("baseline", sme_measure_loop_baseline, groups, 0, repetitions);
  run_probe("fmopa_one_tile", sme_measure_fmopa_one_tile, groups, 1,
            repetitions);
  run_probe("fmopa_four_tiles", sme_measure_fmopa_four_tiles, groups, 4,
            repetitions);
  return 0;
}
