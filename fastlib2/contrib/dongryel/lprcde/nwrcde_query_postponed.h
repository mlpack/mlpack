#ifndef NWRCDE_QUERY_POSTPONED_H
#define NWRCDE_QUERY_POSTPONED_H

#include "nwrcde_delta.h"

class NWRCdeQueryPostponed {
  
 public:
  
  double nwr_numerator_sum_l;
  double nwr_numerator_sum_e;
  double nwr_denominator_sum_l;
  double nwr_denominator_sum_e;
  double nwr_numerator_n_pruned;
  double nwr_denominator_n_pruned;
  double nwr_numerator_used_error;
  double nwr_denominator_used_error;
  
 public:

  void ApplyDelta(const NWRCdeDelta &delta_in) {
    nwr_numerator_sum_l += delta_in.nwr_numerator_sum_l;
    nwr_numerator_sum_e += delta_in.nwr_numerator_sum_e;
    nwr_denominator_sum_l += delta_in.nwr_denominator_sum_l;
    nwr_denominator_sum_e += delta_in.nwr_denominator_sum_e;
    nwr_numerator_n_pruned += delta_in.nwr_numerator_n_pruned;
    nwr_denominator_n_pruned += delta_in.nwr_denominator_n_pruned;
    nwr_numerator_used_error += delta_in.nwr_numerator_used_error;
    nwr_denominator_used_error += delta_in.nwr_denominator_used_error;
  }

  void ApplyPostponed(const NWRCdeQueryPostponed &postponed_in) {
    nwr_numerator_sum_l += postponed_in.nwr_numerator_sum_l;
    nwr_numerator_sum_e += postponed_in.nwr_numerator_sum_e;
    nwr_denominator_sum_l += postponed_in.nwr_denominator_sum_l;
    nwr_denominator_sum_e += postponed_in.nwr_denominator_sum_e;
    nwr_numerator_n_pruned += postponed_in.nwr_numerator_n_pruned;
    nwr_denominator_n_pruned += postponed_in.nwr_denominator_n_pruned;
    nwr_numerator_used_error += postponed_in.nwr_numerator_used_error;
    nwr_denominator_used_error += postponed_in.nwr_denominator_used_error;
  }

  void Init(int num_queries) {
    
    // Reset the postponed quantities to zero.
    SetZero();
  }
  
  void SetZero() {
    nwr_numerator_sum_l = 0;
    nwr_numerator_sum_e = 0;
    nwr_denominator_sum_l = 0;
    nwr_denominator_sum_e = 0;
    nwr_numerator_n_pruned = 0;
    nwr_denominator_n_pruned = 0;
    nwr_numerator_used_error = 0;
    nwr_denominator_used_error = 0;
  }
  
};

#endif
