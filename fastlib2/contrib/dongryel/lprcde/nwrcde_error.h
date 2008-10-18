#ifndef NWRCDE_ERROR_H
#define NWRCDE_ERROR_H

#include "fastlib/fastlib.h"

class NWRCdeError {

 public:
  
  class NWRCdeErrorComponent {
   public:
    double error_per_pair;
    double error;
    
    OT_DEF_BASIC(NWRCdeErrorComponent) {
      OT_MY_OBJECT(error_per_pair);
      OT_MY_OBJECT(error);
    }
  };

  NWRCdeErrorComponent nwr_numerator;
  NWRCdeErrorComponent nwr_denominator;

  OT_DEF_BASIC(NWRCdeError) {
    OT_MY_OBJECT(nwr_numerator);
    OT_MY_OBJECT(nwr_denominator);
  }

 public:

  template<typename TGlobal, typename TQuerySummary, typename ReferenceTree>
  void ComputeAllowableError
  (const TGlobal &parameters,
   const TQuerySummary &new_summary, ReferenceTree *rnode) {

    nwr_numerator.error_per_pair = 
      (parameters.relative_error * new_summary.nwr_numerator_sum_l -
       new_summary.nwr_numerator_used_error_u) /
      (parameters.rset_target_sum - new_summary.nwr_numerator_n_pruned_l);
    nwr_numerator.error = nwr_numerator.error_per_pair *
      rnode->stat().sum_of_target_values;
    
    nwr_denominator.error_per_pair = 
      (parameters.relative_error * new_summary.nwr_denominator_sum_l -
       new_summary.nwr_denominator_used_error_u) /
      (parameters.rset.n_cols() - new_summary.nwr_denominator_n_pruned_l);
    nwr_denominator.error = nwr_denominator.error_per_pair *
      rnode->count();
  }

};

#endif
