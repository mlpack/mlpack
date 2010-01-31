#ifndef NESTED_SUM_UTILITY_H
#define NESTED_SUM_UTILITY_H

#include "strata.h"

class NestedSumUtility {

 public:

  static double ComputeVariance(const Strata &strata, index_t stratum_number) {

    /*
    // For each column, the zero-th row is the sum, and the first row
    // is the squared sum.
    const double current_num_samples = (double)
      strata.num_samples_for_each_stratum[stratum_number];
    const double *statistics = strata.statistics_for_each_stratum.GetColumnPtr
      (stratum_number);
    double current_sample_mean = statistics[0] / current_num_samples;
    
    return (statistics[1] - current_num_samples * math::Sqr(statistics[0])) /
      (current_num_samples - 1.0);
    */
    return 0;
  }

  static void OptimalAllocation(Strata &strata,
				index_t total_num_samples_to_allocate) {

    /*
    // If this is the first time sampling the strata, then assign the
    // same number of samples.
    if(strata.total_num_samples_so_far == 0) {
      for(index_t i = 0; i < strata.total_num_stratum; i++) {
	strata.output_allocation_for_each_stratum[i] =
	  total_samples_to_allocate;
      }

      // Increment the total number of samples used so far.
      strata.total_num_samples_so_far += strata.total_num_stratum *
	total_num_samples_to_allocate;
    }
    else {
      
      double denom = 0.0;
      for(index_t i = 0; i < strata.total_num_stratum; i++) {
	double variance_for_current_stratum = ComputeVariance(strata, i);
	denom += strata.percentage_of_terms_in_each_stratum[i] *
	  sqrt(variance_for_current_stratum);
      }
      for(index_t i = 0; i < strata.total_num_stratum; i++) {
	double variance_for_current_stratum = ComputeVariance(strata, i);
	strata.output_allocation_for_each_stratum[i] = 
	  std::min(1, (index_t) 
		   ceil(total_num_samples_to_allocate * 
			strata.percentage_of_terms_in_each_stratum[i] *
			sqrt(variance_for_current_stratum) / denom));

	strata.total_num_samples_so_far +=
	  strata.output_allocation_for_each_stratum[i];
      }
      
    }
    */

  }
};

#endif
