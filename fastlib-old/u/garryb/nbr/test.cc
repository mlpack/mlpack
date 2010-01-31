#include "fastlib/fastlib.h"
#include <sys/time.h>

int main(int argc, char *argv[]) {
  GaussianStarKernel kernel;
  double mul_factor;

  kernel.Init(0.05, 2);
  mul_factor = kernel.CalcMultiplicativeNormConstant(2);
  
  for (double d = 0; d < 1; d += 0.01) {
    DRange range = kernel.RangeUnnormOnSq(
      DRange(math::Sqr(d - 0.01), math::Sqr(d + 0.01)));

    printf("%f: %f (%f..%f)\n", d, kernel.EvalUnnormOnSq(d*d),
      range.lo, range.hi);
  }

/*  fx_init(argc, argv);
  index_t n = fx_param_int_req(fx_root, "n");
  index_t granularity = fx_param_int(fx_root, "granularity", 8);
  bool fix = fx_param_int(fx_root, "fix", 0);
  struct timeval tv;

  gettimeofday(&tv, NULL);
  srand(tv.tv_usec);

  ArrayList<index_t> a;
  ArrayList<index_t> miss_counts;
  ArrayList<index_t> work;
  index_t total_misses = 0;
  index_t total_work = 0;

  a.Init(n);
  miss_counts.Init(n);
  work.Init(n);
  for (index_t i = 0; i < n; i++) {
    a[i] = i;
    miss_counts[i] = 0;
    //work[i] = (rand() % (granularity * 2)) + 1;
    work[i] = granularity * (n - i - 1) * 2 / n + 1;
    total_work += work[i];
  }

  while (total_work != 0) {
    index_t ready_i = rand() % n;
    index_t orig_victim_i = a[ready_i];
    index_t victim_i = orig_victim_i;
    index_t my_misses = 0;

    while (a[victim_i] != victim_i) {
      victim_i = a[victim_i];
      miss_counts[victim_i]++;
      my_misses++;
    }

    if (fix) {
      index_t victim2_i = orig_victim_i;
      index_t i = my_misses/2;

      while (a[victim2_i] != victim2_i) {
        index_t tmp_i = victim2_i;
        victim2_i = a[victim2_i];
        i--;
        if (i == 0) {
          a[tmp_i] = victim_i;
          miss_counts[tmp_i]++;
          my_misses++;
        }
      }
    }

    // back at the original machine
    a[ready_i] = victim_i;
    miss_counts[ready_i] += my_misses;
    total_misses += my_misses;

    if (victim_i != -1) {
      // at the victim machine
      DEBUG_ASSERT_MSG(work[victim_i] > 0, "%d %d %d", ready_i, victim_i, work[victim_i]);
      work[victim_i]--;
      if (work[victim_i] == 0) {
        //fprintf(stderr, "Out of work: %d from %d\n", victim_i, ready_i);
        a[victim_i] = (victim_i + 1) % n;
      }
      total_work--;
    }
  }

  double sumsq = 0;
  index_t max_miss_count = 0;

  for (index_t i = 0; i < n; i++) {
    index_t my_miss_count = miss_counts[i];
    sumsq += my_miss_count * my_miss_count;
    max_miss_count = max(max_miss_count, my_miss_count);
  }

  fx_format_result(fx_root, "total_misses", "%"LI"d", total_misses);
  fx_format_result(fx_root, "miss_ratio", "%f", 1.0 * total_misses / n);
  fx_format_result(fx_root, "rms", "%f", 1.0 * sqrt(sumsq / n));
  fx_format_result(fx_root, "max_miss_count", "%d", max_miss_count);

  fx_done();
  */
}
