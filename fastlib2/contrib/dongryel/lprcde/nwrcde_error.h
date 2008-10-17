#ifndef NWRCDE_ERROR_H
#define NWRCDE_ERROR_H

#include "fastlib/fastlib.h"

#include "nwrcde_global.h"
#include "nwrcde_query_summary.h"

class NWRCdeError {

 public:
  
  double nwr_numerator_error;
  double nwr_denominator_error;

  OT_DEF_BASIC(NWRCdeError) {
    OT_MY_OBJECT(nwr_numerator_error);
    OT_MY_OBJECT(nwr_denominator_error);
  }

 public:

  template<typename TKernelAux, typename ReferenceTree>
  void ComputeAllowableError
  (const NWRCdeGlobal<TKernelAux, ReferenceTree> &parameters,
   const NWRCdeQuerySummary &new_summary, ReferenceTree *rnode) {

    nwr_numerator_error = 
      (parameters.relative_error * new_summary.nwr_numerator_sum_l -
       new_summary.nwr_numerator_used_error_u) *
      rnode->stat().sum_of_target_values /
      (parameters.rset_target_sum - new_summary.nwr_numerator_n_pruned_l);
    nwr_denominator_error = 
      (parameters.relative_error * new_summary.nwr_denominator_sum_l -
       new_summary.nwr_denominator_used_error_u) * rnode->count() /
      (parameters.rset.n_cols() - new_summary.nwr_denominator_n_pruned_l);
  }

};

#endif
