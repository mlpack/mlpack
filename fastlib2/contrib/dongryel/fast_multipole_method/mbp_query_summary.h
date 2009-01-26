class MultiTreeQuerySummary {
 public:

  DRange negative_potential_bound;

  DRange positive_potential_bound;
    
  double n_pruned_l;
    
  double used_error_u;

  OT_DEF_BASIC(MultiTreeQuerySummary) {
    OT_MY_OBJECT(negative_potential_bound);
    OT_MY_OBJECT(positive_potential_bound);
    OT_MY_OBJECT(n_pruned_l);
    OT_MY_OBJECT(used_error_u);
  }

 public:

  template<typename TQueryResult>
  void Accumulate(const TQueryResult &query_results, index_t q_index) {
    negative_potential_bound |= query_results.
      negative_potential_bound[q_index];
    positive_potential_bound |= query_results.
      positive_potential_bound[q_index];
    n_pruned_l = std::min(n_pruned_l, query_results.n_pruned[q_index]);
    used_error_u = std::max(used_error_u, query_results.used_error[q_index]);
  }

  void SetZero() {
    negative_potential_bound.Init(0, 0);
    positive_potential_bound.Init(0, 0);
    n_pruned_l = 0;
    used_error_u = 0;
  }
    
  void ApplyDelta(const MultiTreeDelta &delta_in, index_t delta_index) {
    negative_potential_bound += 
      delta_in.negative_potential_bound[delta_index];
    positive_potential_bound +=
      delta_in.positive_potential_bound[delta_index];
  }

  void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in) {
    negative_potential_bound += postponed_in.negative_potential_bound;
    positive_potential_bound += postponed_in.positive_potential_bound;
    n_pruned_l += postponed_in.n_pruned;
    used_error_u += postponed_in.used_error;
  }

  void Accumulate(const MultiTreeQuerySummary &summary_in) {
    negative_potential_bound |= summary_in.negative_potential_bound;
    positive_potential_bound |= summary_in.positive_potential_bound;
    n_pruned_l = std::min(n_pruned_l, summary_in.n_pruned_l);
    used_error_u = std::max(used_error_u, summary_in.used_error_u);
  }

  void StartReaccumulate() {
    negative_potential_bound.InitEmptySet();
    positive_potential_bound.InitEmptySet();
    n_pruned_l = DBL_MAX;
    used_error_u = 0;
  }

};
