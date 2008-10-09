#ifndef NWRCDE_DELTA_H
#define NWRCDE_DELTA_H

#include "fastlib/fastlib.h"

class NWRCdeDelta {
  
 public:
  
  DRange dsqd_range;
  DRange kernel_value_range;
  double nwr_numerator_sum_l;
  double nwr_denominator_sum_l;
  double nwr_numerator_sum_e;
  double nwr_denominator_sum_e;
  double nwr_numerator_n_pruned;
  double nwr_denominator_n_pruned;
  double nwr_numerator_used_error;
  double nwr_denominator_used_error;
  
 public:
  
  template<typename Tree, typename TKernel>
  void Compute(Tree *qnode, Tree *rnode, TKernel &kernel) {

    double finite_difference_error;

    dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
    dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
    kernel_value_range = kernel.RangeUnnormOnSq(dsqd_range);

    finite_difference_error = 0.5 * kernel_value_range.width();

    // Compute the bound delta changes based on the kernel value range.
    nwr_numerator_sum_l = rnode->stat().sum_of_target_values * 
      kernel_value_range.lo;
    nwr_denominator_sum_l = rnode->count() * kernel_value_range.lo;
    nwr_numerator_sum_e = rnode->stat().sum_of_target_values *
      kernel_value_range.mid();
    nwr_denominator_sum_e = rnode->count() * kernel_value_range.mid();
    nwr_numerator_n_pruned = rnode->sum_of_target_values;
    nwr_denominator_n_pruned = rnode->count();
    nwr_numerator_used_error = rnode->sum_of_target_values * 
      finite_difference_error;
    nwr_denominator_used_error = rnode->count() * finite_difference_error;
  }

  void SetZero() {
    dsqd_range.Reset(DBL_MAX, -DBL_MAX);
    kernel_value_range.Reset(DBL_MAX, -DBL_MAX);
    nwr_numerator_sum_l = 0;
    nwr_denominator_sum_l = 0;
    nwr_numerator_sum_e = 0;
    nwr_denominator_sum_e = 0;
    nwr_numerator_n_pruned = 0;
    nwr_denominator_n_pruned = 0;
    nwr_numerator_used_error = 0;
    nwr_denominator_used_error = 0;
  }
  
};

#endif
