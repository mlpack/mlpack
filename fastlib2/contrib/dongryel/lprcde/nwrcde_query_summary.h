#ifndef NWRCDE_QUERY_SUMMARY_H
#define NWRCDE_QUERY_SUMMARY_H

#include "fastlib/fastlib.h"

class NWRCdeQuerySummary {
  
 public:
  
  double nwr_numerator_sum_l;
  double nwr_denominator_sum_l;
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

  void Init(int num_queries) {
    
    // Reset the postponed quantities to zero.
    SetZero();
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
  
  void SetZero() {
    nwr_numerator_sum_l = 0;
    nwr_denominator_sum_l = 0;
    nwr_numerator_n_pruned_l = 0;
    nwr_denominator_n_pruned_l = 0;
    nwr_numerator_used_error_u = 0;
    nwr_denominator_used_error_u = 0;
  }
  
};

#endif
