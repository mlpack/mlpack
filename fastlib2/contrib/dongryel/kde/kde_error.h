#ifndef INSIDE_KDE_PROBLEM_H
#error "This is not a public header file!"
#endif

class KdeError {
  
 public:
  
  double error_per_pair;
  double error;
  
  OT_DEF_BASIC(KdeError) {
    OT_MY_OBJECT(error_per_pair);
    OT_MY_OBJECT(error);
  }
  
 public:
  
  template<typename TGlobal, typename TQuerySummary, typename TDelta,
	   typename ReferenceTree>
  void ComputeAllowableError
  (const TGlobal &parameters, const TQuerySummary &new_summary,
   const TDelta &exact_delta, ReferenceTree *rnode) {
    
    error_per_pair = 
      std::max((parameters.relative_error * new_summary.sum_l -
		(new_summary.used_error_u +
		 new_summary.probabilistic_used_error_u)) /
	       (parameters.num_reference_points - new_summary.n_pruned_l), 
	       0.0);
    error = error_per_pair * rnode->count();
  }    
};
