#ifndef NWRCDE_QUERY_SUMMARY_H
#define NWRCDE_QUERY_SUMMARY_H

#include "fastlib/fastlib.h"

class NWRCdeQuerySummary {
  
 public:
  
  double nwr_numerator_sum_l;
  double nwr_denominator_sum_l;
  double n_pruned_l;
  double used_error_u;
  
 public:
  
  void Init(int num_queries) {
    
    // Reset the postponed quantities to zero.
    SetZero();
  }

  void Reset() {
    nwr_numerator_sum_l = DBL_MAX;
    nwr_denominator_sum_l = DBL_MAX;
    n_pruned_l = DBL_MAX;
    used_error_u = 0;
  }
  
  void SetZero() {
    nwr_numerator_sum_l = 0;
    nwr_denominator_sum_l = 0;
    n_pruned_l = 0;
    used_error_u = 0;
  }
  
};

#endif
