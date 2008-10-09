#ifndef NWRCDE_DELTA_H
#define NWRCDE_DELTA_H

#include "fastlib/fastlib.h"

class NWRCdeDelta {
  
 public:
  
  DRange dsqd_range;
  DRange kernel_value_range;
  double nwr_numerator_sum_l;
  double denominator_sum_l;
  double n_pruned;
  double used_error;
  
 public:
  
  template<typename Tree>
  void ComputeDelta(Tree *qnode, Tree *rnode) {
    dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
    dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
    kernel_value_range = ka_.kernel_.RangeUnnormOnSq(dsqd_range);
    n_pruned += rnode->sum_of_target_values_;
  }

  void SetZero() {
    dsqd_range.Reset();
    nwr_numerator_sum_l = 0;
    nwr_denominator_sum_l = 0;
    n_pruned = 0;
  }
  
};

#endif
