#ifndef INSIDE_KDE_PROBLEM_H
#error "This is not a public header file!"
#endif

class MultiTreeQueryResult {

 public:

  Vector sum_l;
  Vector sum_e;
  Vector n_pruned;
  Vector used_error;
  Vector probabilistic_used_error;
    
  int num_finite_difference_prunes;
    
  int num_far_to_local_prunes;
    
  int num_direct_far_prunes;
    
  int num_direct_local_prunes;
    
  /** @brief The estimated Nadaraya-Watson regression estimates,
   *         computed in the postprocessing phrase.
   */
  Vector final_results;
    
  OT_DEF_BASIC(MultiTreeQueryResult) {
    OT_MY_OBJECT(sum_l);
    OT_MY_OBJECT(sum_e);
    OT_MY_OBJECT(n_pruned);
    OT_MY_OBJECT(used_error);
    OT_MY_OBJECT(probabilistic_used_error);
    OT_MY_OBJECT(num_finite_difference_prunes);
    OT_MY_OBJECT(num_far_to_local_prunes);
    OT_MY_OBJECT(num_direct_far_prunes);
    OT_MY_OBJECT(num_direct_local_prunes);
    OT_MY_OBJECT(final_results);
  }
    
 public:

  void Finalize(const MultiTreeGlobal &globals, 
		const ArrayList<index_t> &mapping) {

    MultiTreeUtility::ShuffleAccordingToQueryPermutation(final_results,
							 mapping);
  }

  template<typename TQueryStat>
  void FinalPush(const Matrix &qset, const TQueryStat &query_stat,
		 index_t q_index) {
      
    ApplyPostponed(query_stat.postponed, q_index);
      
    // Evaluate the local expansion.
    sum_e[q_index] +=
      query_stat.local_expansion.EvaluateField(qset, q_index);
  }
    
  template<typename TDelta>
  void ApplyDelta(const TDelta &delta_in, index_t q_index) {
    sum_l[q_index] += delta_in.kde_approximation.sum_l;
    sum_e[q_index] += delta_in.kde_approximation.sum_e;
    n_pruned[q_index] += delta_in.kde_approximation.n_pruned;
    used_error[q_index] += delta_in.kde_approximation.used_error;
    probabilistic_used_error[q_index] =
      sqrt(math::Sqr(probabilistic_used_error[q_index]) +
	   math::Sqr(delta_in.kde_approximation.probabilistic_used_error));
  }

  template<typename TQueryPostponed>
  void ApplyPostponed(const TQueryPostponed &postponed_in,
		      index_t q_index) {
      
    sum_l[q_index] += postponed_in.sum_l;
    sum_e[q_index] += postponed_in.sum_e;
    n_pruned[q_index] += postponed_in.n_pruned;
    used_error[q_index] += postponed_in.used_error;
    probabilistic_used_error[q_index] = 
      sqrt(math::Sqr(probabilistic_used_error[q_index]) +
	   math::Sqr(postponed_in.probabilistic_used_error));
  }

  template<typename ReferenceTree>
  void UpdatePrunedComponents(const ArrayList <ReferenceTree *> &rnodes,
			      index_t q_index) {
    n_pruned[q_index] += rnodes[0]->count();
  }

  void Init(int num_queries) {
      
    // Initialize the space.
    sum_l.Init(num_queries);
    sum_e.Init(num_queries);
    n_pruned.Init(num_queries);
    used_error.Init(num_queries);
    probabilistic_used_error.Init(num_queries);
    final_results.Init(num_queries);
      
    // Reset the sums to zero.
    SetZero();
  }
    
  void PostProcess(const MultiTreeGlobal &globals, index_t q_index) {
    final_results[q_index] = sum_e[q_index] / globals.normalizing_constant;
  }
    
  void PrintDebug(const char *output_file_name) const {
    FILE *stream = fopen(output_file_name, "w+");
      
    for(index_t q = 0; q < final_results.length(); q++) {
      fprintf(stream, "%g\n", final_results[q]);
    }
      
    fclose(stream);
  }
    
  void SetZero() {
    sum_l.SetZero();
    sum_e.SetZero();
    n_pruned.SetZero();
    used_error.SetZero();
    probabilistic_used_error.SetZero();

    num_finite_difference_prunes = 0;
    num_far_to_local_prunes = 0;
    num_direct_far_prunes = 0;
    num_direct_local_prunes = 0;
      
    final_results.SetZero();
  }
};
