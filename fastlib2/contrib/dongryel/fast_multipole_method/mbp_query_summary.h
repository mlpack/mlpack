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

  // Sort the bound quantities.
  static int qsort_compar(const void *a, const void *b) {
    
    double *a_dbl = (double *) a;
    double *b_dbl = (double *) b;
    
    if(*a_dbl < *b_dbl) {
      return -1;
    }
    else if(*a_dbl > *b_dbl) {
      return 1;
    }
    else {
      return 0;
    }
  }

  template<typename TQueryResult>
  void Accumulate(const TQueryResult &query_results, index_t q_index) {
    negative_potential_bound |= query_results.
      negative_potential_bound[q_index];
    positive_potential_bound |= query_results.
      positive_potential_bound[q_index];
    n_pruned_l = std::min(n_pruned_l, query_results.n_pruned[q_index]);
    used_error_u = std::max(used_error_u, query_results.used_error[q_index]);
  }

  template<typename TGlobal, typename TQueryResult>
  void PostAccumulate(TGlobal &globals, const TQueryResult &query_results,
		      index_t first, index_t count) {

    // Sort the negative potentials and get something in the upper
    // quantile.
    for(index_t i = first; i < first + count; i++) {
      globals.tmp_space[i] = query_results.negative_potential_bound[i].hi;
    }
    qsort(globals.tmp_space.ptr() + first, count, sizeof(double), 
	  &qsort_compar);
    negative_potential_bound.hi = 
      globals.tmp_space[std::max(first, (index_t) 
				 (first + count - 1 - 
				  globals.percentile * count))];

    // Sort the positive potentials and get something in the lower
    // quantile.
    for(index_t i = first; i < first + count; i++) {
      globals.tmp_space[i] = query_results.positive_potential_bound[i].lo;
    }
    qsort(globals.tmp_space.ptr() + first, count, sizeof(double), 
	  &qsort_compar);
    positive_potential_bound.lo = 
      globals.tmp_space[std::min((index_t)
				 (first + globals.percentile * count),
				  first + count - 1)];

    // Sort the pruned counts and get something in the lower quantile.
    for(index_t i = first; i < first + count; i++) {
      globals.tmp_space[i] = query_results.n_pruned[i];
    }
    qsort(globals.tmp_space.ptr() + first, count, sizeof(double),
	  &qsort_compar);
    n_pruned_l = 
      globals.tmp_space[std::min((index_t)
				 (first + globals.percentile * count),
				 first + count - 1)];

    // Sort the used error and get something in the upper quantile.
    for(index_t i = first; i < first + count; i++) {
      globals.tmp_space[i] = query_results.used_error[i];
    }
    qsort(globals.tmp_space.ptr() + first, count, sizeof(double),
	  &qsort_compar);
    used_error_u = globals.tmp_space[std::max(first, 
					      (index_t)
					      (first + count - 1 -
					       globals.percentile * count))];
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
