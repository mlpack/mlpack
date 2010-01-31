#ifndef NWRCDE_PROBLEM_H
#define NWRCDE_PROBLEM_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "../multitree_template/multitree_utility.h"

template<typename TKernelAux>
class NWRCdeProblem {

 public:

  static const int num_hybrid_sets = 0;

  static const int num_query_sets = 1;

  static const int num_reference_sets = 1;

  static const int order = 2;

  class MultiTreeDelta {
    
   public:
    
    /** @brief The approximation method types.
     */
    enum ApproximationType {
      
      /** @brief Exhaustive method.
       */
      EXHAUSTIVE,
      
      /** @brief Finite difference method.
       */
      FINITE_DIFFERENCE,
      
      /** @brief Far-to-local translation method.
       */
      FAR_TO_LOCAL,
      
      /** @brief Direct far-field evaluation method.
       */
      DIRECT_FARFIELD,
      
      /** @brief Direct local accumulation method.
       */
      DIRECT_LOCAL
    };
    
    class NWRCdeApproximation {
      
     public:
      
      /** @brief The decided approximation type.
       */
      ApproximationType approx_type;
      
      int order_farfield_to_local;
      
      int order_farfield;
      
      int order_local;
      
      int cost_farfield_to_local;
      
      int cost_farfield;
      
      int cost_local;
      
      int cost_exhaustive;
      
      int min_cost;
      
      double actual_err_farfield_to_local;
      
      double actual_err_farfield;
      
      double actual_err_local;
      
      double sum_l;
      double sum_e;
      double n_pruned;
      double used_error;
      
      OT_DEF_BASIC(NWRCdeApproximation) {
	OT_MY_OBJECT(approx_type);
	OT_MY_OBJECT(order_farfield_to_local);
	OT_MY_OBJECT(order_farfield);
	OT_MY_OBJECT(order_local);
	OT_MY_OBJECT(cost_farfield_to_local);
	OT_MY_OBJECT(cost_farfield);
	OT_MY_OBJECT(cost_local);
	OT_MY_OBJECT(cost_exhaustive);
	OT_MY_OBJECT(min_cost);
	OT_MY_OBJECT(actual_err_farfield_to_local);
	OT_MY_OBJECT(actual_err_farfield);
	OT_MY_OBJECT(actual_err_local);
	OT_MY_OBJECT(sum_l);
	OT_MY_OBJECT(sum_e);
	OT_MY_OBJECT(n_pruned);
	OT_MY_OBJECT(used_error);
      }
      
     public:
      
      template<typename TGlobal, typename QueryTree, typename ReferenceTree>
      void Init(const TGlobal &parameters, QueryTree *qnode,
		ReferenceTree *rnode) {
	
	// Initialize the member variables. The threshold exhaustive
	// cost is half of what is for single summation.
	Reset();
	cost_exhaustive = parameters.dimension * qnode->count() * 
	  rnode->count() / 2;
      }
      
      void Reset() {
	approx_type = EXHAUSTIVE;
	order_farfield_to_local = -1;
	order_farfield = -1;
	order_local = -1;
	cost_farfield_to_local = INT_MAX;
	cost_farfield = INT_MAX;
	cost_local = INT_MAX;
	cost_exhaustive = INT_MAX;
	min_cost = INT_MAX;
	actual_err_farfield_to_local = DBL_MAX;
	actual_err_farfield = DBL_MAX;
	actual_err_local = DBL_MAX;
	sum_l = 0;
	sum_e = 0;
	n_pruned = 0;
	used_error = 0;
      }      
    };

    DRange dsqd_range;
    DRange kernel_value_range;
    NWRCdeApproximation nwr_numerator;
    NWRCdeApproximation nwr_denominator;
    
    OT_DEF_BASIC(MultiTreeDelta) {
      OT_MY_OBJECT(dsqd_range);
      OT_MY_OBJECT(kernel_value_range);
      OT_MY_OBJECT(nwr_numerator);
      OT_MY_OBJECT(nwr_denominator);
    }
    
   public:
    
    template<typename TFarFieldExpansion, typename TLocalExpansion,
	     typename TGlobal, typename QueryTree, typename ReferenceTree, 
	     typename TErrorComponent>
    void DecideSeriesApproximationMethod
    (const TGlobal &parameters,
     QueryTree *qnode, TLocalExpansion &local_expansion, ReferenceTree *rnode,
     TFarFieldExpansion &farfield_expansion,
     const TErrorComponent &allowed_error,
     NWRCdeApproximation &approximation) {
      
      approximation.order_farfield_to_local =
	farfield_expansion.OrderForConvertingToLocal
	(rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi, 
	 allowed_error.error_per_pair,
	 &(approximation.actual_err_farfield_to_local));
      approximation.order_farfield = 
	farfield_expansion.OrderForEvaluating
	(rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi,
	 allowed_error.error_per_pair,
	 &(approximation.actual_err_farfield));
      approximation.order_local = 
	local_expansion.OrderForEvaluating
	(rnode->bound(), qnode->bound(), dsqd_range.lo, dsqd_range.hi,
	 allowed_error.error_per_pair, 
	 &(approximation.actual_err_local));
      
      if(approximation.order_farfield_to_local >= 0) {
	approximation.cost_farfield_to_local = (int) 
	  parameters.kernel_aux.sea_.FarFieldToLocalTranslationCost
	  (approximation.order_farfield_to_local);
      }
      if(approximation.order_farfield >= 0) {
	approximation.cost_farfield = (int) 
	  parameters.kernel_aux.sea_.FarFieldEvaluationCost
	  (approximation.order_farfield) * (qnode->count());
      }
      if(approximation.order_local >= 0) {
	approximation.cost_local = (int) 
	  parameters.kernel_aux.sea_.DirectLocalAccumulationCost
	  (approximation.order_local) * (rnode->count());
      }    

      approximation.min_cost = min(approximation.cost_farfield_to_local, 
				   min(approximation.cost_farfield, 
				       min(approximation.cost_local,
					   approximation.cost_exhaustive)));
      
      // Decide which method to use.
      if(approximation.cost_farfield_to_local == approximation.min_cost) {
	
	approximation.approx_type = FAR_TO_LOCAL;
	approximation.used_error = farfield_expansion.get_weight_sum() * 
	  approximation.actual_err_farfield_to_local;
	approximation.n_pruned = farfield_expansion.get_weight_sum();
	approximation.order_farfield = approximation.order_local = -1;
	approximation.sum_e = 0;
      }
      
      else if(approximation.cost_farfield == approximation.min_cost) {
	
	approximation.approx_type = DIRECT_FARFIELD;
	approximation.used_error = farfield_expansion.get_weight_sum() * 
	  approximation.actual_err_farfield;
	approximation.n_pruned = farfield_expansion.get_weight_sum();
	approximation.order_farfield_to_local = approximation.order_local = -1;
	approximation.sum_e = 0;
      }
      
      else if(approximation.cost_local == approximation.min_cost) {
	
	approximation.approx_type = DIRECT_LOCAL;
	approximation.used_error = farfield_expansion.get_weight_sum() * 
	  approximation.actual_err_local;
	approximation.n_pruned = farfield_expansion.get_weight_sum();
	approximation.order_farfield_to_local = 
	  approximation.order_farfield = -1;
	approximation.sum_e = 0;
      }
    }
    
    template<typename TGlobal, typename QueryTree, typename ReferenceTree,
	     typename TError>
    bool ComputeSeriesExpansion
    (const TGlobal &parameters, QueryTree *qnode, ReferenceTree *rnode,
     const TError &allowed_error) {
      
      // Decide methods for both the numerator and the denominator.
      DecideSeriesApproximationMethod
	(parameters, qnode, qnode->stat().nwr_numerator_local_expansion, rnode,
	 rnode->stat().nwr_numerator_farfield_expansion,
	 allowed_error.nwr_numerator, nwr_numerator);
      DecideSeriesApproximationMethod
	(parameters, qnode, qnode->stat().nwr_denominator_local_expansion,
	 rnode, rnode->stat().nwr_denominator_farfield_expansion,
	 allowed_error.nwr_denominator, nwr_denominator);
      
      return (nwr_numerator.approx_type != EXHAUSTIVE) &&
	(nwr_denominator.approx_type != EXHAUSTIVE);
    }
    
    template<typename TGlobal, typename QueryTree, typename ReferenceTree>
    void ComputeFiniteDifference
    (const TGlobal &parameters, QueryTree *qnode, ReferenceTree *rnode) {
      
      double finite_difference_error;
      
      dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
      dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
      kernel_value_range = 
	parameters.kernel_aux.kernel_.RangeUnnormOnSq(dsqd_range);
      
      finite_difference_error = 0.5 * kernel_value_range.width();
      
      // Compute the bound delta changes based on the kernel value range.
      nwr_numerator.sum_l = rnode->stat().sum_of_target_values * 
	kernel_value_range.lo;
      nwr_denominator.sum_l = rnode->count() * kernel_value_range.lo;
      nwr_numerator.sum_e = rnode->stat().sum_of_target_values *
	kernel_value_range.mid();
      nwr_denominator.sum_e = rnode->count() * kernel_value_range.mid();
      nwr_numerator.n_pruned = rnode->stat().sum_of_target_values;
      nwr_denominator.n_pruned = rnode->count();
      nwr_numerator.used_error = rnode->stat().sum_of_target_values * 
	finite_difference_error;
      nwr_denominator.used_error = rnode->count() * finite_difference_error;
    }
    
    template<typename TGlobal, typename QueryTree, typename ReferenceTree>
    void Reset(const TGlobal &parameters, QueryTree *qnode,
	       ReferenceTree *rnode) {
      
      dsqd_range.Reset(DBL_MAX, -DBL_MAX);
      kernel_value_range.Reset(DBL_MAX, -DBL_MAX);
      nwr_numerator.Init(parameters, qnode, rnode);
      nwr_denominator.Init(parameters, qnode, rnode);
    }    
  };

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
    (const TGlobal &parameters, const TQuerySummary &new_summary,
     ReferenceTree *rnode) {

      nwr_numerator.error_per_pair = 
	(parameters.relative_error * new_summary.nwr_numerator_sum_l -
	 new_summary.nwr_numerator_used_error_u) /
	(parameters.rset_target_sum[0] - new_summary.nwr_numerator_n_pruned_l);
      nwr_numerator.error = nwr_numerator.error_per_pair *
	rnode->stat().sum_of_target_values;
      
      nwr_denominator.error_per_pair = 
	(parameters.relative_error * new_summary.nwr_denominator_sum_l -
	 new_summary.nwr_denominator_used_error_u) /
	(parameters.num_reference_points -
	 new_summary.nwr_denominator_n_pruned_l);
      nwr_denominator.error = nwr_denominator.error_per_pair *
	rnode->count();
    }    
  };

  class MultiTreeGlobal {

   public:

    /** @brief The pointer to the module holding the parameters.
     */
    struct datanode *module;
    
    /** @brief The dimensionality.
     */
    int dimension;

    /** @brief The kernel function.
     */
    TKernelAux kernel_aux;
    
    /** @brief The relative error desired.
     */
    double relative_error;

    /** @brief The reference vector target sums.
     */
    Vector rset_target_sum;

    /** @brief The number of reference points.
     */
    int num_reference_points;

    ArrayList<int> hybrid_node_chosen_indices;
    ArrayList<int> query_node_chosen_indices;
    ArrayList<int> reference_node_chosen_indices;
    
    // It is important not to include the module pointer because it will
    // be freed by fx_done()!
    OT_DEF_BASIC(MultiTreeGlobal) {
      OT_MY_OBJECT(dimension);
      OT_MY_OBJECT(kernel_aux);
      OT_MY_OBJECT(relative_error);
      OT_MY_OBJECT(rset_target_sum);
      OT_MY_OBJECT(hybrid_node_chosen_indices);
      OT_MY_OBJECT(query_node_chosen_indices);
      OT_MY_OBJECT(reference_node_chosen_indices);
    }
    
   public:

    void Init(int num_queries, int dimension_in, 
	      const ArrayList<Matrix *> &targets, struct datanode *module_in) {
      
      // Set the data node module to incoming one.
      module = module_in;

      hybrid_node_chosen_indices.Init(NWRCdeProblem::num_hybrid_sets);
      query_node_chosen_indices.Init(NWRCdeProblem::num_query_sets);
      reference_node_chosen_indices.Init(NWRCdeProblem::num_reference_sets);
      
      // Set the dimension.
      dimension = dimension_in;

      // Initialize the bandwidth.
      double bandwidth = fx_param_double_req(module, "bandwidth");

      // Initialize the series expansion object.
      if(dimension <= 2) {
	kernel_aux.Init(bandwidth, fx_param_int(module, "order", 7),
			dimension);
      }
      else if(dimension <= 3) {
	kernel_aux.Init(bandwidth, fx_param_int(module, "order", 5),
			dimension);
      }
      else if(dimension <= 5) {
	kernel_aux.Init(bandwidth, fx_param_int(module, "order", 3),
			dimension);
      }
      else if(dimension <= 6) {
	kernel_aux.Init(bandwidth, fx_param_int(module, "order", 1),
			dimension);
      }
      else {
	kernel_aux.Init(bandwidth, fx_param_int(module, "order", 0),
			dimension);
      }

      // Set the relative error.
      relative_error = fx_param_double(module, "relative_error", 0.1);
      
      // Compute the reference target sums.
      const Matrix &reference_targets = *(targets[0]);
      rset_target_sum.Init(reference_targets.n_cols());
      rset_target_sum.SetZero();
      for(index_t c = 0; c < reference_targets.n_cols(); c++) {	
	for(index_t r = 0; r < reference_targets.n_rows(); r++) {
	  rset_target_sum[c] += reference_targets.get(r, c);
	}
      }
      
      // Set the number of reference points.
      num_reference_points = reference_targets.n_rows();
      
    }

  };

  class MultiTreeQueryResult {

   public:

    Vector nwr_numerator_sum_l;
    Vector nwr_numerator_sum_e;
    Vector nwr_denominator_sum_l;
    Vector nwr_denominator_sum_e;
    Vector nwr_numerator_n_pruned;
    Vector nwr_denominator_n_pruned;
    Vector n_pruned;
    Vector nwr_numerator_used_error;
    Vector nwr_denominator_used_error;
    
    int num_finite_difference_prunes;
    
    int num_far_to_local_prunes;
    
    int num_direct_far_prunes;
    
    int num_direct_local_prunes;
    
    /** @brief The estimated Nadaraya-Watson regression estimates,
     *         computed in the postprocessing phrase.
     */
    Vector final_results;
    
    OT_DEF_BASIC(MultiTreeQueryResult) {
      OT_MY_OBJECT(nwr_numerator_sum_l);
      OT_MY_OBJECT(nwr_numerator_sum_e);
      OT_MY_OBJECT(nwr_denominator_sum_l);
      OT_MY_OBJECT(nwr_denominator_sum_e);
      OT_MY_OBJECT(nwr_numerator_n_pruned);
      OT_MY_OBJECT(nwr_denominator_n_pruned);
      OT_MY_OBJECT(n_pruned);
      OT_MY_OBJECT(nwr_numerator_used_error);
      OT_MY_OBJECT(nwr_denominator_used_error);
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
      nwr_numerator_sum_e[q_index] +=
	query_stat.nwr_numerator_local_expansion.EvaluateField(qset, q_index);
      nwr_denominator_sum_e[q_index] +=
	query_stat.nwr_denominator_local_expansion.EvaluateField(qset,
								 q_index);
    }
    
    template<typename TQueryPostponed>
    void ApplyPostponed(const TQueryPostponed &postponed_in,
			index_t q_index) {
      
      nwr_numerator_sum_l[q_index] += postponed_in.nwr_numerator_sum_l;
      nwr_numerator_sum_e[q_index] += postponed_in.nwr_numerator_sum_e;
      nwr_denominator_sum_l[q_index] += postponed_in.nwr_denominator_sum_l;
      nwr_denominator_sum_e[q_index] += postponed_in.nwr_denominator_sum_e;
      nwr_numerator_n_pruned[q_index] += postponed_in.nwr_numerator_n_pruned;
      nwr_denominator_n_pruned[q_index] += 
	postponed_in.nwr_denominator_n_pruned;
      nwr_numerator_used_error[q_index] +=
	postponed_in.nwr_numerator_used_error;
      nwr_denominator_used_error[q_index] +=
	postponed_in.nwr_denominator_used_error;
    }

    template<typename ReferenceTree>
    void UpdatePrunedComponents(const ArrayList <ReferenceTree *> &rnodes,
				index_t q_index) {
      nwr_numerator_n_pruned[q_index] += rnodes[0]->stat().
	sum_of_target_values;
      nwr_denominator_n_pruned[q_index] += rnodes[0]->count();
    }

    void Init(int num_queries) {
      
      // Initialize the space.
      nwr_numerator_sum_l.Init(num_queries);
      nwr_numerator_sum_e.Init(num_queries);
      nwr_denominator_sum_l.Init(num_queries);
      nwr_denominator_sum_e.Init(num_queries);
      nwr_numerator_n_pruned.Init(num_queries);
      nwr_denominator_n_pruned.Init(num_queries);
      n_pruned.Init(num_queries);
      nwr_numerator_used_error.Init(num_queries);
      nwr_denominator_used_error.Init(num_queries);      
      final_results.Init(num_queries);
      
      // Reset the sums to zero.
      SetZero();
    }
    
    template<typename MultiTreeGlobal>
    void PostProcess(const MultiTreeGlobal &globals, index_t q_index) {
      
      if(nwr_denominator_sum_e[q_index] == 0 &&
	 nwr_numerator_sum_e[q_index] == 0) {
	final_results[q_index] = 0;
      }
      else {
	final_results[q_index] = nwr_numerator_sum_e[q_index] /
	  nwr_denominator_sum_e[q_index];
      }
    }
    
    void PrintDebug(const char *output_file_name) const {
      FILE *stream = fopen(output_file_name, "w+");
      
      for(index_t q = 0; q < final_results.length(); q++) {
	fprintf(stream, "%g %g %g\n", final_results[q],
		nwr_numerator_n_pruned[q], nwr_denominator_n_pruned[q]);
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
      n_pruned.SetZero();
      nwr_numerator_used_error.SetZero();
      nwr_denominator_used_error.SetZero();

      num_finite_difference_prunes = 0;
      num_far_to_local_prunes = 0;
      num_direct_far_prunes = 0;
      num_direct_local_prunes = 0;
      
      final_results.SetZero();
    }
  };

  class MultiTreeQueryPostponed {
    
   public:
  
    double nwr_numerator_sum_l;
    double nwr_numerator_sum_e;
    double nwr_denominator_sum_l;
    double nwr_denominator_sum_e;
    double nwr_numerator_n_pruned;
    double nwr_denominator_n_pruned;
    double nwr_numerator_used_error;
    double nwr_denominator_used_error;
    
   public:

    template<typename TDelta>
    void ApplyDelta(const TDelta &delta_in) {
      nwr_numerator_sum_l += delta_in.nwr_numerator.sum_l;
      nwr_numerator_sum_e += delta_in.nwr_numerator.sum_e;
      nwr_denominator_sum_l += delta_in.nwr_denominator.sum_l;
      nwr_denominator_sum_e += delta_in.nwr_denominator.sum_e;
      nwr_numerator_n_pruned += delta_in.nwr_numerator.n_pruned;
      nwr_denominator_n_pruned += delta_in.nwr_denominator.n_pruned;
      nwr_numerator_used_error += delta_in.nwr_numerator.used_error;
      nwr_denominator_used_error += delta_in.nwr_denominator.used_error;
    }
    
    template<typename TQueryPostponed>
    void ApplyPostponed(const TQueryPostponed &postponed_in) {
      nwr_numerator_sum_l += postponed_in.nwr_numerator_sum_l;
      nwr_numerator_sum_e += postponed_in.nwr_numerator_sum_e;
      nwr_denominator_sum_l += postponed_in.nwr_denominator_sum_l;
      nwr_denominator_sum_e += postponed_in.nwr_denominator_sum_e;
      nwr_numerator_n_pruned += postponed_in.nwr_numerator_n_pruned;
      nwr_denominator_n_pruned += postponed_in.nwr_denominator_n_pruned;
      nwr_numerator_used_error += postponed_in.nwr_numerator_used_error;
      nwr_denominator_used_error += postponed_in.nwr_denominator_used_error;
    }
    
    void SetZero() {
      nwr_numerator_sum_l = 0;
      nwr_numerator_sum_e = 0;
      nwr_denominator_sum_l = 0;
      nwr_denominator_sum_e = 0;
      nwr_numerator_n_pruned = 0;
      nwr_denominator_n_pruned = 0;
      nwr_numerator_used_error = 0;
      nwr_denominator_used_error = 0;
    }
    
  };

  class MultiTreeQuerySummary {
    
   public:
    
    /** @brief The lower bound on the NWR numerator sum.
     */
    double nwr_numerator_sum_l;
    
    /** @brief The lower bound on the NWR denominator sum.
     */
    double nwr_denominator_sum_l;
    
    /** @brief The lower bound on the portion pruned for the NWR
     *         numerator sum.
     */
    double nwr_numerator_n_pruned_l;
    
    double nwr_denominator_n_pruned_l;
    
    double nwr_numerator_used_error_u;
    
    double nwr_denominator_used_error_u;
    
    OT_DEF_BASIC(MultiTreeQuerySummary) {
      OT_MY_OBJECT(nwr_numerator_sum_l);
      OT_MY_OBJECT(nwr_denominator_sum_l);
      OT_MY_OBJECT(nwr_numerator_n_pruned_l);
      OT_MY_OBJECT(nwr_denominator_n_pruned_l);
      OT_MY_OBJECT(nwr_numerator_used_error_u);
      OT_MY_OBJECT(nwr_denominator_used_error_u);
    }
    
   public:
    
    void Init() {
      
      // Reset the postponed quantities to zero.
      SetZero();
    }
    
    template<typename TQueryResult>
    void Accumulate(const TQueryResult &query_results, index_t q_index) {
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
    
    template<typename TQuerySummary>
    void Accumulate(const TQuerySummary &other_summary_results) {
      nwr_numerator_sum_l = std::min
	(nwr_numerator_sum_l, other_summary_results.nwr_numerator_sum_l);
      nwr_denominator_sum_l = 
	std::min(nwr_denominator_sum_l, 
		 other_summary_results.nwr_denominator_sum_l);
      nwr_numerator_n_pruned_l = 
	std::min(nwr_numerator_n_pruned_l,
		 other_summary_results.nwr_numerator_n_pruned_l);
      nwr_denominator_n_pruned_l =
	std::min(nwr_denominator_n_pruned_l,
		 other_summary_results.nwr_denominator_n_pruned_l);
      nwr_numerator_used_error_u =
	std::max(nwr_numerator_used_error_u,
		 other_summary_results.nwr_numerator_used_error_u);
      nwr_denominator_used_error_u = 
	std::max(nwr_denominator_used_error_u,
		 other_summary_results.nwr_denominator_used_error_u);
    }
    
    template<typename TDelta>
    void ApplyDelta(const TDelta &delta_in) {
      nwr_numerator_sum_l += delta_in.nwr_numerator.sum_l;
      nwr_denominator_sum_l += delta_in.nwr_denominator.sum_l;
    }
    
    template<typename TQueryPostponed>
    void ApplyPostponed(const TQueryPostponed &postponed_in) {
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

  class MultiTreeQueryStat {
   public:

    MultiTreeQueryPostponed postponed;
    
    MultiTreeQuerySummary summary;
    
    typename TKernelAux::TLocalExpansion nwr_numerator_local_expansion;
    
    typename TKernelAux::TLocalExpansion nwr_denominator_local_expansion;
    
    OT_DEF_BASIC(MultiTreeQueryStat) {
      OT_MY_OBJECT(postponed);
      OT_MY_OBJECT(summary);
      OT_MY_OBJECT(nwr_numerator_local_expansion);
      OT_MY_OBJECT(nwr_denominator_local_expansion);
    }

   public:

    void FinalPush(MultiTreeQueryStat &child_stat) {
      child_stat.postponed.ApplyPostponed(postponed);
      nwr_numerator_local_expansion.TranslateToLocal
	(child_stat.nwr_numerator_local_expansion);
      nwr_denominator_local_expansion.TranslateToLocal
	(child_stat.nwr_denominator_local_expansion);
    }
    
    void SetZero() {
      postponed.SetZero();
      summary.SetZero();
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const MultiTreeQueryStat& left_stat,
	      const MultiTreeQueryStat& right_stat) {
    }
    
    void Init(const TKernelAux &kernel_aux_in) {
      nwr_numerator_local_expansion.Init(kernel_aux_in);
      nwr_denominator_local_expansion.Init(kernel_aux_in);
    }
    
    template<typename TBound>
    void Init(const TBound &bounding_primitive,
	      const TKernelAux &kernel_aux_in) {
      
      // Initialize the center of expansions and bandwidth for series
      // expansion.
      Vector bounding_box_center;
      Init(kernel_aux_in);
      bounding_primitive.CalculateMidpoint(&bounding_box_center);
      (nwr_numerator_local_expansion.get_center())->CopyValues
	(bounding_box_center);
      (nwr_denominator_local_expansion.get_center())->CopyValues
	(bounding_box_center);
      
      // Reset the postponed quantities to zero.
      SetZero();
    }
  };

  class MultiTreeReferenceStat {
   public:

    /** @brief The far field expansion for the numerator created by the
     *         reference points in this node.
     */
    typename TKernelAux::TFarFieldExpansion nwr_numerator_farfield_expansion;
    
    /** @brief The far field expansion for the denominator created by
     *         the reference points in this node.
     */
    typename TKernelAux::TFarFieldExpansion nwr_denominator_farfield_expansion;
    
    double sum_of_target_values;
    
    OT_DEF_BASIC(MultiTreeReferenceStat) {
      OT_MY_OBJECT(nwr_numerator_farfield_expansion);
      OT_MY_OBJECT(nwr_denominator_farfield_expansion);
      OT_MY_OBJECT(sum_of_target_values);
    }
    
   public:
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const MultiTreeReferenceStat& left_stat,
	      const MultiTreeReferenceStat& right_stat) {
      
    }
    
    template<typename TBound>
    void PostInitCommon(const TBound &bounding_primitive,
			const TKernelAux &kernel_aux_in) {
      
      // Initialize the center of expansions and bandwidth for series
      // expansion.
      Vector bounding_box_center;
      nwr_numerator_farfield_expansion.Init(kernel_aux_in);
      nwr_denominator_farfield_expansion.Init(kernel_aux_in);
      bounding_primitive.CalculateMidpoint(&bounding_box_center);
      (nwr_numerator_farfield_expansion.get_center())->CopyValues
	(bounding_box_center);
      (nwr_denominator_farfield_expansion.get_center())->CopyValues
	(bounding_box_center);
    }
    
    /** @brief Computes the sum of the target values owned by the
     *         reference statistics for a leaf node.
     */
    template<typename TBound>
    void PostInit(const TBound &bounding_primitive,
		  const TKernelAux &kernel_aux_in,
		  const ArrayList<Matrix *> &reference_sets,
		  const ArrayList<Matrix *> &targets,
		  index_t start, index_t count) {
      
      PostInitCommon(bounding_primitive, kernel_aux_in);
      
      // Exhaustively compute multipole moments.
      const Matrix &reference_set = *(reference_sets[0]);
      const Matrix &targets_dereferenced = *(targets[0]);
      Vector nwr_numerator_weights_alias;
      Vector nwr_denominator_weights_alias;
      targets_dereferenced.MakeColumnVector(0, &nwr_numerator_weights_alias);
      targets_dereferenced.MakeColumnVector(1, &nwr_denominator_weights_alias);
      nwr_numerator_farfield_expansion.AccumulateCoeffs
	(reference_set, nwr_numerator_weights_alias, start, start + count,
	 kernel_aux_in.sea_.get_max_order());
      nwr_denominator_farfield_expansion.AccumulateCoeffs
	(reference_set, nwr_denominator_weights_alias, start, start + count,
	 kernel_aux_in.sea_.get_max_order());
      
      sum_of_target_values = 0;
      for(index_t i = start; i < start + count; i++) {
	sum_of_target_values += nwr_numerator_weights_alias[i];	
      }
    }
    
    /** @brief Computes the sum of the target values owned by the
     *         reference statistics for an internal node.
     */
    template<typename TBound>
    void PostInit(const TBound &bounding_primitive,
		  const TKernelAux &kernel_aux_in,
		  const ArrayList<Matrix *> &reference_sets,
		  const ArrayList<Matrix *> &targets,
		  index_t start, index_t count,
		  const MultiTreeReferenceStat& left_stat,
		  const MultiTreeReferenceStat& right_stat) {
      
      PostInitCommon(bounding_primitive, kernel_aux_in);
      
      // Translate the moments up from the two children's moments.
      nwr_numerator_farfield_expansion.TranslateFromFarField
	(left_stat.nwr_numerator_farfield_expansion);
      nwr_numerator_farfield_expansion.TranslateFromFarField
	(right_stat.nwr_numerator_farfield_expansion);
      nwr_denominator_farfield_expansion.TranslateFromFarField
	(left_stat.nwr_denominator_farfield_expansion);
      nwr_denominator_farfield_expansion.TranslateFromFarField
	(right_stat.nwr_denominator_farfield_expansion);
      
      sum_of_target_values = left_stat.sum_of_target_values +
	right_stat.sum_of_target_values;
    }
    
  };

  template<typename TGlobal, typename QueryTree, typename ReferenceTree,
	   typename TQueryResult, typename TDelta>
  static void ApplySeriesExpansion(const TGlobal &globals, const Matrix &qset,
				   const Matrix &rset,
				   const Matrix &reference_weights,
				   QueryTree *qnode, ReferenceTree *rnode, 
				   TQueryResult &query_results,
				   const TDelta &delta) {

    // FIX ME for multi-target NWR code!
    Vector nwr_numerator_weights_alias;
    Vector nwr_denominator_weights;
    reference_weights.MakeColumnVector(0, &nwr_numerator_weights_alias);
    reference_weights.MakeColumnVector(1, &nwr_denominator_weights);

    switch(delta.nwr_numerator.approx_type) {
      case TDelta::FAR_TO_LOCAL:
	DEBUG_ASSERT(delta.nwr_numerator.order_farfield_to_local >= 0);
	rnode->stat().nwr_numerator_farfield_expansion.TranslateToLocal
	  (qnode->stat().nwr_numerator_local_expansion, 
	   delta.nwr_numerator.order_farfield_to_local);
	query_results.num_far_to_local_prunes++;
	break;
      case TDelta::DIRECT_FARFIELD:
	DEBUG_ASSERT(delta.nwr_numerator.order_farfield >= 0);
	for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	  query_results.nwr_numerator_sum_e[q] += 
	    rnode->stat().nwr_numerator_farfield_expansion.EvaluateField
	    (qset, q, delta.nwr_numerator.order_farfield);
	}
	query_results.num_direct_far_prunes++;
	break;
      case TDelta::DIRECT_LOCAL:
	DEBUG_ASSERT(delta.nwr_numerator.order_local >= 0);
	qnode->stat().nwr_numerator_local_expansion.AccumulateCoeffs
	  (rset, nwr_numerator_weights_alias, rnode->begin(), 
	   rnode->end(), delta.nwr_numerator.order_local);
	query_results.num_direct_local_prunes++;
	break;
      default:
	break;
    }
    switch(delta.nwr_denominator.approx_type) {
      case TDelta::FAR_TO_LOCAL:
	DEBUG_ASSERT(delta.nwr_denominator.order_farfield_to_local >= 0);
	rnode->stat().nwr_denominator_farfield_expansion.TranslateToLocal
	  (qnode->stat().nwr_denominator_local_expansion, 
	   delta.nwr_denominator.order_farfield_to_local);
	query_results.num_far_to_local_prunes++;
	break;
      case TDelta::DIRECT_FARFIELD:
	DEBUG_ASSERT(delta.nwr_denominator.order_farfield >= 0);
	for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	  query_results.nwr_denominator_sum_e[q] += 
	    rnode->stat().nwr_denominator_farfield_expansion.EvaluateField
	    (qset, q, delta.nwr_denominator.order_farfield);
	}
	query_results.num_direct_far_prunes++;
	break;
      case TDelta::DIRECT_LOCAL:
	DEBUG_ASSERT(delta.nwr_denominator.order_local >= 0);
	qnode->stat().nwr_denominator_local_expansion.AccumulateCoeffs
	  (rset, nwr_denominator_weights, rnode->begin(),
	   rnode->end(), delta.nwr_denominator.order_local);
	query_results.num_direct_local_prunes++;
	break;
      default:
	break;
    }

    // Lastly, we need to add on the rest of the delta contributions.
    qnode->stat().postponed.ApplyDelta(delta);
  }

  template<typename MultiTreeGlobal, typename MultiTreeQueryResult,
	   typename HybridTree, typename QueryTree, typename ReferenceTree>
  static bool ConsiderTupleExact(MultiTreeGlobal &globals,
				 MultiTreeQueryResult &query_results,
				 MultiTreeDelta &delta,
				 const ArrayList<Matrix *> &query_sets,
				 const ArrayList<Matrix *> &reference_sets,
				 const ArrayList<Matrix *> &targets,
				 ArrayList<HybridTree *> &hybrid_nodes,
				 ArrayList<QueryTree *> &query_nodes,
				 ArrayList<ReferenceTree *> &reference_nodes,
				 double total_num_tuples,
				 double total_n_minus_one_tuples_root,
				 const Vector &total_n_minus_one_tuples) {

    // Query and reference nodes.
    QueryTree *qnode = query_nodes[0];
    ReferenceTree *rnode = reference_nodes[0];
    
    // Initialize the delta.
    delta.Reset(globals, qnode, rnode);

    // Initialize the query summary.
    MultiTreeQuerySummary new_summary;
    new_summary.Init();

    // Compute the bound changes due to a finite-difference
    // approximation.
    delta.ComputeFiniteDifference(globals, qnode, rnode);

    // Refine the lower bound using the new lower bound info.
    new_summary.InitCopy(qnode->stat().summary);
    new_summary.ApplyPostponed(qnode->stat().postponed);
    new_summary.ApplyDelta(delta);
    
    // Compute the allowable error.
    typename NWRCdeProblem::NWRCdeError allowed_error;
    allowed_error.ComputeAllowableError(globals, new_summary, rnode);

    // If the error bound is satisfied by the hard error bound, it is
    // safe to prune.
    if((!isnan(allowed_error.nwr_numerator.error)) && 
       (!isnan(allowed_error.nwr_denominator.error))) {
      
      if((delta.nwr_numerator.used_error <= allowed_error.nwr_numerator.error)
	 && (delta.nwr_denominator.used_error <= 
	     allowed_error.nwr_denominator.error)) {

	qnode->stat().postponed.ApplyDelta(delta);
	query_results.num_finite_difference_prunes++; 
	return true;
      }

      // Try series expansion.
      else if(delta.ComputeSeriesExpansion(globals, qnode, rnode, 
					   allowed_error)) {
	NWRCdeProblem::ApplySeriesExpansion(globals, *(query_sets[0]),
					    *(reference_sets[0]),
					    *(targets[0]), qnode,
					    rnode, query_results, delta);
	return true;
      }

      else {
	return false;
      }
    }
    else {
      return false;
    }

  }
  
  template<typename MultiTreeGlobal, typename MultiTreeQueryResult,
	   typename HybridTree, typename QueryTree, typename ReferenceTree>
  static bool ConsiderTupleProbabilistic
  (MultiTreeGlobal &globals, MultiTreeQueryResult &results,
   MultiTreeDelta &exact_delta,
   const ArrayList<Matrix *> &query_sets,
   const ArrayList<Matrix *> &reference_sets,
   ArrayList<HybridTree *> &hybrid_nodes,
   ArrayList<QueryTree *> &query_nodes,
   ArrayList<ReferenceTree *> &reference_nodes,
   double total_num_tuples,
   double total_n_minus_one_tuples_root,
   const Vector &total_n_minus_one_tuples) {

    return false;
  }

  static void HybridNodeEvaluateMain(MultiTreeGlobal &globals,
				     const ArrayList<Matrix *> &query_sets,
				     const ArrayList<Matrix *> &sets,
				     const ArrayList<Matrix *> &targets,
				     MultiTreeQueryResult &query_results) {
    
  }

  static void ReferenceNodeEvaluateMain(MultiTreeGlobal &globals,
					const ArrayList<Matrix *> &query_sets,
					const ArrayList<Matrix *> &sets,
					const ArrayList<Matrix *> &targets,
					MultiTreeQueryResult &query_results) {
    
    // Compute pairwise contributions...
    index_t q_index = globals.query_node_chosen_indices[0];
    index_t r_index = globals.reference_node_chosen_indices[0];

    double squared_distance =
      la::DistanceSqEuclidean(globals.dimension,
			      query_sets[0]->GetColumnPtr(q_index),
			      sets[0]->GetColumnPtr(r_index));
    double kernel_value = globals.kernel_aux.kernel_.EvalUnnormOnSq
      (squared_distance);
    double weighted_kernel_value = kernel_value * targets[0]->get(r_index, 0);

    query_results.nwr_numerator_sum_l[q_index] += weighted_kernel_value;
    query_results.nwr_numerator_sum_e[q_index] += weighted_kernel_value;
    query_results.nwr_denominator_sum_l[q_index] += kernel_value;
    query_results.nwr_denominator_sum_e[q_index] += kernel_value;
  }
};

#endif
