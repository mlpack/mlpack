#ifndef NWRCDE_QUERY_POSTPONED_H
#define NWRCDE_QUERY_POSTPONED_H

#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/mult_farfield_expansion.h"
#include "mlpack/series_expansion/mult_local_expansion.h"
#include "mlpack/series_expansion/kernel_aux.h"

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

  template<typename TDelta>
  void ApplyDelta(const TDelta &delta_in) {
    nwr_numerator_sum_l += delta_in.nwr_numerator.sum_l;
    nwr_numerator_sum_e += delta_in.nwr_numerator.sum_e;
    nwr_denominator_sum_l += delta_in.nwr_denominator.sum_l;
    nwr_denominator_sum_e += delta_in.nwr_denominator.sum_e;
    nwr_numerator_n_pruned += delta_in.nwr_numerator.n_pruned;
    nwr_denominator_n_pruned += delta_in.nwr_denominator.n_pruned;
    nwr_numerator_used_error += delta_in.nwr_numerator.used_error;
    nwr_denominator_used_error += delta_in.nwr_denominator.used_error;
  }

  template<typename TQueryPostponed>
  void ApplyPostponed(const TQueryPostponed &postponed_in) {
    nwr_numerator_sum_l += postponed_in.nwr_numerator_sum_l;
    nwr_numerator_sum_e += postponed_in.nwr_numerator_sum_e;
    nwr_denominator_sum_l += postponed_in.nwr_denominator_sum_l;
    nwr_denominator_sum_e += postponed_in.nwr_denominator_sum_e;
    nwr_numerator_n_pruned += postponed_in.nwr_numerator_n_pruned;
    nwr_denominator_n_pruned += postponed_in.nwr_denominator_n_pruned;
    nwr_numerator_used_error += postponed_in.nwr_numerator_used_error;
    nwr_denominator_used_error += postponed_in.nwr_denominator_used_error;
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
