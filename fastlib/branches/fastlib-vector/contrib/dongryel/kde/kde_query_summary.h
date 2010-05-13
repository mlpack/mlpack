#ifndef INSIDE_KDE_PROBLEM_H
#error "This is not a public header file!"
#endif

class MultiTreeQuerySummary {
    
 public:
    
  double sum_l;
    
  double n_pruned_l;

  double n_pruned_u;

  double used_error_u;
    
  double probabilistic_used_error_u;
    
  OT_DEF_BASIC(MultiTreeQuerySummary) {
    OT_MY_OBJECT(sum_l);
    OT_MY_OBJECT(n_pruned_l);
    OT_MY_OBJECT(n_pruned_u);
    OT_MY_OBJECT(used_error_u);
    OT_MY_OBJECT(probabilistic_used_error_u);
  }
  
 public:
    
  void Init() {
      
    // Reset the postponed quantities to zero.
    SetZero();
  }
    
  template<typename TQueryResult>
  void Init(const TQueryResult &query_results, index_t q_index) {
    sum_l = query_results.sum_l[q_index];
    n_pruned_l = query_results.n_pruned[q_index];
    n_pruned_u = query_results.n_pruned[q_index];
    used_error_u = query_results.used_error[q_index];
    probabilistic_used_error_u = query_results.probabilistic_used_error
      [q_index];
  }

  template<typename TQueryResult>
  void Accumulate(const TQueryResult &query_results, index_t q_index) {

    sum_l = std::min(sum_l, query_results.sum_l[q_index]);
    n_pruned_l = std::min(n_pruned_l, query_results.n_pruned[q_index]);
    n_pruned_u = std::max(n_pruned_u, query_results.n_pruned[q_index]);
    used_error_u = std::max(used_error_u, query_results.used_error[q_index]);
    probabilistic_used_error_u =
      std::max(probabilistic_used_error_u,
	       query_results.probabilistic_used_error[q_index]);
  }
    
  template<typename TQuerySummary>
  void Accumulate(const TQuerySummary &other_summary_results) {

    sum_l = std::min(sum_l, other_summary_results.sum_l);
    n_pruned_l = std::min(n_pruned_l, other_summary_results.n_pruned_l);
    n_pruned_u = std::max(n_pruned_u, other_summary_results.n_pruned_u);
    used_error_u = std::max(used_error_u,
			    other_summary_results.used_error_u);
    probabilistic_used_error_u =
      std::max(probabilistic_used_error_u,
	       other_summary_results.probabilistic_used_error_u);
  }
    
  template<typename TDelta>
  void ApplyDelta(const TDelta &delta_in) {

    sum_l += delta_in.kde_approximation.sum_l;
  }
    
  template<typename TQueryPostponed>
  void ApplyPostponed(const TQueryPostponed &postponed_in) {
      
    sum_l += postponed_in.sum_l;
    n_pruned_l += postponed_in.n_pruned;
    n_pruned_u += postponed_in.n_pruned;
    used_error_u += postponed_in.used_error;
    probabilistic_used_error_u = 
      sqrt(math::Sqr(probabilistic_used_error_u) +
	   math::Sqr(postponed_in.probabilistic_used_error));
  }
    
  void StartReaccumulate() {

    sum_l = DBL_MAX;
    n_pruned_l = DBL_MAX;
    n_pruned_u = 0;
    used_error_u = 0;
    probabilistic_used_error_u = 0;
  }
    
  void SetZero() {

    sum_l = 0;
    n_pruned_l = 0;
    n_pruned_u = 0;
    used_error_u = 0;
    probabilistic_used_error_u = 0;
  }
    
};
