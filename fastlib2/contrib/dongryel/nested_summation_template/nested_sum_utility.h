#ifndef NESTED_SUM_UTILITY_H
#define NESTED_SUM_UTILITY_H

class NestedSumUtility {

 public:

  static double ComputeVariance
  (const ArrayList<index_t> &num_samples_for_each_strata,
   const Matrix &statistics_for_each_strata, index_t strata_number) {

    // For each column, the zero-th row is the sum, and the first row
    // is the squared sum.
    const double current_num_samples = (double)
      num_samples_for_each_strata[strata_number];
    const double *statistics = statistics_for_each_strata.GetColumnPtr
      (strata_number);
    double current_sample_mean = statistics[0] / current_num_samples;
    
    return (statistics[1] - current_num_samples * math::Sqr(statistics[0])) /
      (current_num_samples - 1.0);
  }

  static void OptimalAllocation
  (const ArrayList<index_t> &num_samples_for_each_strata,
   index_t &total_num_samples_so_far, const Matrix &statistics_for_each_strata,
   index_t total_samples_to_allocate,
   ArrayList<index_t> &output_allocation_for_each_strata) {

    if(total_num_samples_so_far == 0) {
      for(index_t i = 0; i < output_allocation_for_each_strata; i++) {
	output_allocation_for_each_strata[i] = total_samples_to_allocate;
      }
      total_num_samples_so_far = num_samples_for_each_strata.size() *
	total_samples_to_allocate;
    }
    else {
    }

  }
};

#endif
