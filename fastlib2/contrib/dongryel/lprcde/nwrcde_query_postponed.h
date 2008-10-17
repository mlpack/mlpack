#ifndef NWRCDE_QUERY_POSTPONED_H
#define NWRCDE_QUERY_POSTPONED_H

#include "nwrcde_delta.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/mult_farfield_expansion.h"
#include "mlpack/series_expansion/mult_local_expansion.h"
#include "mlpack/series_expansion/kernel_aux.h"

template<typename TKernelAux>
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
 
  /** @brief The far field expansion created by the reference points
   *         in this node.
   */
  typename TKernelAux::TFarFieldExpansion farfield_expansion;
  
  /** @brief The local expansion stored in this node.
   */
  typename TKernelAux::TLocalExpansion local_expansion;

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

  void Init(const TKernelAux &kernel_aux_in) {
    farfield_expansion.Init(kernel_aux_in);
    local_expansion.Init(kernel_aux_in);
  }

  template<typename TBound>
  void Init(const TBound &bounding_primitive,
	    const TKernelAux &kernel_aux_in) {
 
    // Initialize the center of expansions and bandwidth for series
    // expansion.
    Vector bounding_box_center;
    Init(kernel_aux_in);
    bounding_primitive.CalculateMidpoint(&bounding_box_center);
    (farfield_expansion.get_center())->CopyValues(bounding_box_center);
    (local_expansion.get_center())->CopyValues(bounding_box_center);
   
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
