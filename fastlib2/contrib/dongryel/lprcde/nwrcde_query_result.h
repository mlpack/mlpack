#ifndef NWRCDE_QUERY_RESULT_H
#define NWRCDE_QUERY_RESULT_H

#include "nwrcde_query_postponed.h"

class NWRCdeQueryResult {
  
 public:

  Vector nwr_numerator_sum_l;
  Vector nwr_numerator_sum_e;
  Vector nwr_denominator_sum_l;
  Vector nwr_denominator_sum_e;
  Vector nwr_numerator_n_pruned;
  Vector nwr_denominator_n_pruned;
  Vector nwr_numerator_used_error;
  Vector nwr_denominator_used_error;
  
 public:
  
  void ApplyPostponed(const NWRCdeQueryPostponed &postponed_in, 
		      index_t q_index) {
    
    nwr_numerator_sum_l[q_index] += postponed_in.nwr_numerator_sum_l;
    nwr_numerator_sum_e[q_index] += postponed_in.nwr_numerator_sum_e;
    nwr_denominator_sum_l[q_index] += postponed_in.nwr_denominator_sum_l;
    nwr_denominator_sum_e[q_index] += postponed_in.nwr_denominator_sum_e;
    nwr_numerator_n_pruned[q_index] += postponed_in.nwr_numerator_n_pruned;
    nwr_denominator_n_pruned[q_index] += postponed_in.nwr_denominator_n_pruned;
    nwr_numerator_used_error[q_index] += postponed_in.nwr_numerator_used_error;
    nwr_denominator_used_error[q_index] +=
      postponed_in.nwr_denominator_used_error;
  }

  void Init(int num_queries) {
    
    // Initialize the space.
    nwr_numerator_sum_l.Init(num_queries);
    nwr_numerator_sum_e.Init(num_queries);
    nwr_denominator_sum_l.Init(num_queries);
    nwr_denominator_sum_e.Init(num_queries);
    nwr_numerator_n_pruned.Init(num_queries);
    nwr_denominator_n_pruned.Init(num_queries);
    nwr_numerator_used_error.Init(num_queries);
    nwr_denominator_used_error.Init(num_queries);
    
    // Reset the sums to zero.
    SetZero();
  }
  
  void SetZero() {
    nwr_numerator_sum_l.SetZero();
    nwr_numerator_sum_e.SetZero();
    nwr_denominator_sum_l.SetZero();
    nwr_denominator_sum_e.SetZero();
    nwr_numerator_n_pruned.SetZero();
    nwr_denominator_n_pruned.SetZero();
    nwr_numerator_used_error.SetZero();
    nwr_denominator_used_error.SetZero();
  }
  
};

#endif
