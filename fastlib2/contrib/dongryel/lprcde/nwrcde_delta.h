#ifndef NWRCDE_DELTA_H
#define NWRCDE_DELTA_H

#include "fastlib/fastlib.h"

class NWRCdeDelta {
  
 public:
  
  double nwr_numerator_sum_l;
  double denominator_sum_l;
  double n_pruned_l;
  
 public:
  
  void Init(int num_queries) {
    
    // Reset the delta quantities to zero.
    SetZero();
  }
  
  void SetZero() {
    nwr_numerator_sum_l = 0;
    nwr_denominator_sum_l = 0;
    n_pruned_l = 0;
  }
  
};

#endif
