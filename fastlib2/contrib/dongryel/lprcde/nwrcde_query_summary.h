#ifndef NWRCDE_QUERY_SUMMARY_H
#define NWRCDE_QUERY_SUMMARY_H

#include "fastlib/fastlib.h"

#include "nwrcde_query_result.h"
#include "nwrcde_global.h"

class NWRCdeQuerySummary {
  
 public:
  
  /** @brief The lower bound on the NWR numerator sum.
   */
  double nwr_numerator_sum_l;

  /** @brief The lower bound on the NWR denominator sum.
   */
  double nwr_denominator_sum_l;

  /** @brief The lower bound on the portion pruned for the NWR
   *         numerator sum.
   */
  double nwr_numerator_n_pruned_l;

  double nwr_denominator_n_pruned_l;

  double nwr_numerator_used_error_u;

  double nwr_denominator_used_error_u;

  OT_DEF_BASIC(NWRCdeQuerySummary) {
    OT_MY_OBJECT(nwr_numerator_sum_l);
    OT_MY_OBJECT(nwr_denominator_sum_l);
    OT_MY_OBJECT(nwr_numerator_n_pruned_l);
    OT_MY_OBJECT(nwr_denominator_n_pruned_l);
    OT_MY_OBJECT(nwr_numerator_used_error_u);
    OT_MY_OBJECT(nwr_denominator_used_error_u);
  }

 public:

  template<typename TKernel, typename ReferenceTree>
  void Init(const NWRCdeGlobal<TKernel, ReferenceTree> &globals) {
    
    // Reset the postponed quantities to zero.
    SetZero(globals);
  }

  void Accumulate(const NWRCdeQueryResult &query_results, index_t q_index) {
    nwr_numerator_sum_l = 
      std::min(nwr_numerator_sum_l,
	       query_results.nwr_numerator_sum_l[q_index]);
    nwr_denominator_sum_l = 
      std::min(nwr_denominator_sum_l,
	       query_results.nwr_denominator_sum_l[q_index]);
    nwr_numerator_n_pruned_l =
      std::min(nwr_numerator_n_pruned_l,
	       query_results.nwr_numerator_n_pruned[q_index]);
    nwr_denominator_n_pruned_l =
      std::min(nwr_denominator_n_pruned_l,
	       query_results.nwr_denominator_n_pruned[q_index]);
    nwr_numerator_used_error_u =
      std::max(nwr_numerator_used_error_u,
	       query_results.nwr_numerator_used_error[q_index]);
    nwr_denominator_used_error_u =
      std::max(nwr_denominator_used_error_u,
	       query_results.nwr_denominator_used_error[q_index]);
  }

  void Accumulate(const NWRCdeQuerySummary &other_summary_results) {
    nwr_numerator_sum_l = std::min(nwr_numerator_sum_l,
				   other_summary_results.nwr_numerator_sum_l);
    nwr_denominator_sum_l = 
      std::min(nwr_denominator_sum_l, 
	       other_summary_results.nwr_denominator_sum_l);
    nwr_numerator_n_pruned_l = 
      std::min(nwr_numerator_n_pruned_l,
	       other_summary_results.nwr_numerator_n_pruned_l);
    nwr_denominator_n_pruned_l =
      std::min(nwr_denominator_n_pruned_l,
	       other_summary_results.nwr_denominator_n_pruned_l);
    nwr_numerator_used_error_u =
      std::max(nwr_numerator_used_error_u,
	       other_summary_results.nwr_numerator_used_error_u);
    nwr_denominator_used_error_u = 
      std::max(nwr_denominator_used_error_u,
	       other_summary_results.nwr_denominator_used_error_u);
  }

  void ApplyDelta(const NWRCdeDelta &delta_in) {
    nwr_numerator_sum_l += delta_in.nwr_numerator_sum_l;
    nwr_denominator_sum_l += delta_in.nwr_denominator_sum_l;
  }

  void ApplyPostponed(const NWRCdeQueryPostponed &postponed_in) {
    nwr_numerator_sum_l += postponed_in.nwr_numerator_sum_l;
    nwr_denominator_sum_l += postponed_in.nwr_denominator_sum_l;
    nwr_numerator_n_pruned_l += postponed_in.nwr_numerator_n_pruned;
    nwr_denominator_n_pruned_l += postponed_in.nwr_denominator_n_pruned;
    nwr_numerator_used_error_u += postponed_in.nwr_numerator_used_error;
    nwr_denominator_used_error_u += postponed_in.nwr_denominator_used_error;
  }

  void StartReaccumulate() {
    nwr_numerator_sum_l = DBL_MAX;
    nwr_denominator_sum_l = DBL_MAX;
    nwr_numerator_n_pruned_l = DBL_MAX;
    nwr_denominator_n_pruned_l = DBL_MAX;
    nwr_numerator_used_error_u = 0;
    nwr_denominator_used_error_u = 0;
  }
  
  template<typename TKernel, typename ReferenceTree>
  void SetZero(const NWRCdeGlobal<TKernel, ReferenceTree> &globals) {
    nwr_numerator_sum_l = 0;
    nwr_denominator_sum_l = 0;
    nwr_numerator_n_pruned_l = 0;
    nwr_denominator_n_pruned_l = 0;
    nwr_numerator_used_error_u = 0;
    nwr_denominator_used_error_u = 0;
  }
  
};

#endif
