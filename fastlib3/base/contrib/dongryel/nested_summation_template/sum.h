#ifndef SUM_H
#define SUM_H

#include "fastlib/fastlib.h"
#include "operator.h"
#include "nested_sum_utility.h"

class Sum: public Operator {

 public:

  double NaiveCompute(std::map<index_t, index_t> &constant_dataset_indices) {

    double sum_result = 0;
 
    // Get the list of restrictions associated with the current
    // dataset index.
    std::map<index_t, std::vector<int> >::iterator restriction = 
      restrictions_->find(dataset_index_);
  
    // The current dataset that is involved.
    const Matrix *dataset = (*datasets_)[dataset_index_];

    // Loop over all indices of the current operator.
    for(index_t n = 0; n < dataset->n_cols(); n++) {

      if(restriction != restrictions_->end() &&
	 CheckViolation_(constant_dataset_indices, (*restriction).second, n)) {
	continue;
      }

      // Re-assign the point index for the current dataset.
      constant_dataset_indices[dataset_index_] = n;

      // Recursively evaluate the operators and add them up.
      for(index_t i = 0; i < operators_.size(); i++) {
	sum_result += operators_[i]->NaiveCompute(constant_dataset_indices);
      }      
    }
    return PostProcess_(constant_dataset_indices, sum_result);
  }

  double MonteCarloCompute
  (ArrayList<Strata> &list_of_strata,
   std::map<index_t, index_t> &constant_dataset_indices,
   double relative_error, double probability) {

    double sum_result = 0;

    // Sample over some combinations of the current operator, and
    // recursively call the operators underneath the current operator,
    // and summing all of them up.
    int total_num_samples_needed = Operator::min_num_samples_;
    
    
    do {

      // Do sample allocations over the strata.
      NestedSumUtility::OptimalAllocation(list_of_strata[dataset_index_],
					  total_num_samples_needed);

      for(index_t strata_index = 0;
	  strata_index < list_of_strata[dataset_index_].total_num_stratum;
	  strata_index++) {

	for(index_t s = 0; s < total_num_samples_needed; s++) {
	  
	  ChoosePointIndex_(constant_dataset_indices);
	  
	  // Recursively evaluate the operators and add them up.
	  for(index_t i = 0; i < operators_.size(); i++) {
	    sum_result += operators_[i]->MonteCarloCompute
	      (list_of_strata, constant_dataset_indices, relative_error,
	       probability);
	  }
	}
      }

      // Recompute the threshold and the samples needed for the next
      // iteration...

      /*
      // FIX ME!!!!!
      double standard_score = 0;
      int threshold = math::Sqr(standard_score * (1 + relative_error) /
				(relative_error * sample_mean)) *
	sample_variance;
      */
      
    } while(total_num_samples_needed > 0);

    /*
    // Compute the sample average and multiply by the number of terms
    // in the current dataset index.
    sum_result = sum_result / ((double) sample_size) * 
      (((*datasets_)[dataset_index_])->n_cols());
    
    return PostProcess_(constant_dataset_indices, sum_result);
    */
    return 0;
  }

};

#endif
