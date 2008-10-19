#ifndef NWRCDE_QUERY_RESULT_H
#define NWRCDE_QUERY_RESULT_H

class NWRCdeQueryResult {
  
 public:

  Vector nwr_numerator_sum_l;
  Vector nwr_numerator_sum_e;
  Vector nwr_denominator_sum_l;
  Vector nwr_denominator_sum_e;
  Vector nwr_numerator_n_pruned;
  Vector nwr_denominator_n_pruned;
  Vector nwr_numerator_used_error;
  Vector nwr_denominator_used_error;
  
  int num_finite_difference_prunes;

  int num_far_to_local_prunes;
  
  int num_direct_far_prunes;
  
  int num_direct_local_prunes;

  /** @brief The estimated Nadaraya-Watson regression estimates,
   *         computed in the postprocessing phrase.
   */
  Vector final_nwr_estimates;

  OT_DEF_BASIC(NWRCdeQueryResult) {
    OT_MY_OBJECT(nwr_numerator_sum_l);
    OT_MY_OBJECT(nwr_numerator_sum_e);
    OT_MY_OBJECT(nwr_denominator_sum_l);
    OT_MY_OBJECT(nwr_denominator_sum_e);
    OT_MY_OBJECT(nwr_numerator_n_pruned);
    OT_MY_OBJECT(nwr_denominator_n_pruned);
    OT_MY_OBJECT(nwr_numerator_used_error);
    OT_MY_OBJECT(nwr_denominator_used_error);
    OT_MY_OBJECT(num_finite_difference_prunes);
    OT_MY_OBJECT(num_far_to_local_prunes);
    OT_MY_OBJECT(num_direct_far_prunes);
    OT_MY_OBJECT(num_direct_local_prunes);
  }

 public:
  
  template<typename TQueryStat>
  void FinalPush(const Matrix &qset, const TQueryStat &query_stat, 
		 index_t q_index) {
    
    ApplyPostponed(query_stat.postponed, q_index);
    
    // Evaluate the local expansion.
    nwr_numerator_sum_e[q_index] += 
      query_stat.nwr_numerator_local_expansion.EvaluateField(qset, q_index);
    nwr_denominator_sum_e[q_index] +=
      query_stat.nwr_denominator_local_expansion.EvaluateField(qset, q_index);
  }

  template<typename TQueryPostponed>
  void ApplyPostponed(const TQueryPostponed &postponed_in, 
		      index_t q_index) {
    
    nwr_numerator_sum_l[q_index] += postponed_in.nwr_numerator_sum_l;
    nwr_numerator_sum_e[q_index] += postponed_in.nwr_numerator_sum_e;
    nwr_denominator_sum_l[q_index] += postponed_in.nwr_denominator_sum_l;
    nwr_denominator_sum_e[q_index] += postponed_in.nwr_denominator_sum_e;
    nwr_numerator_n_pruned[q_index] += postponed_in.nwr_numerator_n_pruned;
    nwr_denominator_n_pruned[q_index] += postponed_in.nwr_denominator_n_pruned;
    nwr_numerator_used_error[q_index] += postponed_in.nwr_numerator_used_error;
    nwr_denominator_used_error[q_index] +=
      postponed_in.nwr_denominator_used_error;
  }

  void Init(int num_queries) {
    
    // Initialize the space.
    nwr_numerator_sum_l.Init(num_queries);
    nwr_numerator_sum_e.Init(num_queries);
    nwr_denominator_sum_l.Init(num_queries);
    nwr_denominator_sum_e.Init(num_queries);
    nwr_numerator_n_pruned.Init(num_queries);
    nwr_denominator_n_pruned.Init(num_queries);
    nwr_numerator_used_error.Init(num_queries);
    nwr_denominator_used_error.Init(num_queries);

    final_nwr_estimates.Init(num_queries);
    
    // Reset the sums to zero.
    SetZero();
  }

  void Postprocess(index_t q_index) {
    
    if(nwr_denominator_sum_e[q_index] == 0 &&
       nwr_numerator_sum_e[q_index] == 0) {
      final_nwr_estimates[q_index] = 0;
    }
    else {
      final_nwr_estimates[q_index] = nwr_numerator_sum_e[q_index] /
	nwr_denominator_sum_e[q_index];
    }
  }

  void PrintDebug(const char *output_file_name) const {
    FILE *stream = fopen(output_file_name, "w+");

    for(index_t q = 0; q < final_nwr_estimates.length(); q++) {
      fprintf(stream, "%g\n", final_nwr_estimates[q]);
    }
    
    fclose(stream);
  }

  void SetZero() {
    nwr_numerator_sum_l.SetZero();
    nwr_numerator_sum_e.SetZero();
    nwr_denominator_sum_l.SetZero();
    nwr_denominator_sum_e.SetZero();
    nwr_numerator_n_pruned.SetZero();
    nwr_denominator_n_pruned.SetZero();
    nwr_numerator_used_error.SetZero();
    nwr_denominator_used_error.SetZero();
    num_finite_difference_prunes = 0;
    num_far_to_local_prunes = 0;
    num_direct_far_prunes = 0;
    num_direct_local_prunes = 0;
    
    final_nwr_estimates.SetZero();
  }
  
};

#endif
