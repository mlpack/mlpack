class MultiTreeQueryResult {
 public:

  ArrayList<DRange> negative_potential_bound;

  Vector negative_potential_e;

  ArrayList<DRange> positive_potential_bound;

  Vector positive_potential_e;

  Vector final_results;

  Vector n_pruned;
    
  Vector used_error;

  /** @brief The number of finite-difference prunes.
   */
  int num_finite_difference_prunes;
  
  int num_monte_carlo_prunes;

  OT_DEF_BASIC(MultiTreeQueryResult) {
    OT_MY_OBJECT(negative_potential_bound);
    OT_MY_OBJECT(negative_potential_e);
    OT_MY_OBJECT(positive_potential_bound);
    OT_MY_OBJECT(positive_potential_e);
    OT_MY_OBJECT(final_results);
    OT_MY_OBJECT(n_pruned);
    OT_MY_OBJECT(used_error);
    OT_MY_OBJECT(num_finite_difference_prunes);
    OT_MY_OBJECT(num_monte_carlo_prunes);
  }

 public:

  void MaximumRelativeError(const MultiTreeQueryResult &other_results,
			    double required_relative_error,
			    double *max_relative_error,
			    double *negative_max_relative_error,
			    double *positive_max_relative_error) {

    int num_over_relative_error = 0;
    int num_over_positive_relative_error = 0;
    int num_over_negative_relative_error = 0;

    double max_absolute_error = 0;
    *max_relative_error = 0;
    *negative_max_relative_error = 0;
    *positive_max_relative_error = 0;

    const char *error_output_filename = 
      fx_param_str(fx_root, "error_output", "error.txt");
    FILE *error_output = fopen(error_output_filename, "w+");

    for(index_t i = 0; i < final_results.length(); i++) {
      
      double relative_error, negative_relative_error, positive_relative_error;
      
      positive_relative_error = (positive_potential_e[i] == 
				 other_results.positive_potential_e[i]) ?
	0:( fabs(positive_potential_e[i] - 
		 other_results.positive_potential_e[i]) /
	    positive_potential_e[i]);
      negative_relative_error = (negative_potential_e[i] == 
				 other_results.negative_potential_e[i]) ?
	0:( fabs(negative_potential_e[i] - 
		 other_results.negative_potential_e[i]) /
	    fabs(negative_potential_e[i]));
      relative_error = (final_results[i] == other_results.final_results[i]) ?
	0:( fabs(final_results[i] - other_results.final_results[i]) /
	    fabs(final_results[i]));
      
      *max_relative_error = std::max(*max_relative_error, relative_error);
      *negative_max_relative_error = std::max(*negative_max_relative_error,
					      negative_relative_error);
      *positive_max_relative_error = std::max(*positive_max_relative_error,
					      positive_relative_error);

      if(relative_error > required_relative_error) {
	num_over_relative_error++;
      }
      if(positive_relative_error > required_relative_error) {
	num_over_positive_relative_error++;
      }
      if(negative_relative_error > required_relative_error) {
	num_over_negative_relative_error++;
      }
      max_absolute_error = std::max(max_absolute_error,
				    fabs(final_results[i] - 
					 other_results.final_results[i]));

      fprintf(error_output, "%g %g %g %g\n", relative_error,
	      negative_relative_error, positive_relative_error,
	      fabs(final_results[i] - other_results.final_results[i]));
    }
    printf("%d particles violated the relative error...", 
	   num_over_relative_error);
    printf("%d particles violated the relative error in the positive...",
	   num_over_positive_relative_error);
    printf("%d particles violated the relative error in the negative...",
	   num_over_negative_relative_error);
    printf("The maximum absolute error was %g...", max_absolute_error);
    fclose(error_output);
  }

  template<typename Tree>
  void UpdatePrunedComponents(const ArrayList<Tree *> &reference_nodes,
			      index_t q_index) {
  }

  void FinalPush(const Matrix &qset, 
		 const MultiTreeQueryStat &stat_in, index_t q_index) {
      
    ApplyPostponed(stat_in.postponed, q_index);
  }

  void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in, 
		      index_t q_index) {

    negative_potential_bound[q_index] += 
      postponed_in.negative_potential_bound;
    negative_potential_e[q_index] += postponed_in.negative_potential_e;
    positive_potential_bound[q_index] +=
      postponed_in.positive_potential_bound;
    positive_potential_e[q_index] += postponed_in.positive_potential_e;
    n_pruned[q_index] += postponed_in.n_pruned;
    used_error[q_index] += postponed_in.used_error;
  }

  void Init(int num_queries) {
    negative_potential_bound.Init(num_queries);
    negative_potential_e.Init(num_queries);
    positive_potential_bound.Init(num_queries);
    positive_potential_e.Init(num_queries);
    final_results.Init(num_queries);
    n_pruned.Init(num_queries);
    used_error.Init(num_queries);
      
    SetZero();
  }

  template<typename MultiTreeGlobal>
  void PostProcess(const MultiTreeGlobal &globals, index_t q_index) {

    final_results[q_index] = negative_potential_e[q_index] +
      positive_potential_e[q_index];
  }

  template<typename MultiTreeGlobal>
  void Finalize(const MultiTreeGlobal &globals,
		const ArrayList<index_t> &mapping) {

    MultiTreeUtility::ShuffleAccordingToQueryPermutation(positive_potential_e,
							 mapping);
    MultiTreeUtility::ShuffleAccordingToQueryPermutation(negative_potential_e,
							 mapping);
    MultiTreeUtility::ShuffleAccordingToQueryPermutation(final_results,
							 mapping);
  }

  void PrintDebug(const char *output_file_name) const {

    FILE *output_file = fopen(output_file_name, "w+");
    
    for(index_t i = 0; i < final_results.length(); i++) {
      fprintf(output_file, "%g %g\n", final_results[i], n_pruned[i]);
    }
    fclose(output_file);
  }

  void SetZero() {

    for(index_t i = 0; i < negative_potential_bound.size(); i++) {
      negative_potential_bound[i].Init(0, 0);
      positive_potential_bound[i].Init(0, 0);
    }
    negative_potential_e.SetZero();
    positive_potential_e.SetZero();
    final_results.SetZero();
    n_pruned.SetZero();
    used_error.SetZero();

    num_finite_difference_prunes = 0;
    num_monte_carlo_prunes = 0;
  }
};
