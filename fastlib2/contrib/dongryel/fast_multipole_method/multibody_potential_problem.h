#ifndef MULTIBODY_POTENTIAL_PROBLEM_H
#define MULTIBODY_POTENTIAL_PROBLEM_H

#include "fastlib/fastlib.h"

#include "mlpack/series_expansion/kernel_aux.h"

#include "at_potential_kernel.h"
#include "three_body_gaussian_kernel.h"
#include "../multitree_template/multitree_utility.h"

template<typename TKernel>
class MultibodyPotentialProblem {

 public:

  class MultiTreeDelta {

   public:

    /** @brief Stores the negative lower and the negative upper
     *         contribution of the $i$-th node in consideration among
     *         the $n$ tuples.
     */
    ArrayList<DRange> negative_potential_bound;

    /** @brief The estimated negative component.
     */
    Vector negative_potential_e;

    /** @brief Stores the positive lower and the positive upper
     *         contribution of the $i$-th node in consideration among
     *         the $n$ tuples.
     */    
    ArrayList<DRange> positive_potential_bound;

    /** @brief The estimated positive component.
     */
    Vector positive_potential_e;

    Vector n_pruned;
    
    Vector used_error;

    OT_DEF_BASIC(MultiTreeDelta) {
      OT_MY_OBJECT(negative_potential_bound);
      OT_MY_OBJECT(negative_potential_e);
      OT_MY_OBJECT(positive_potential_bound);
      OT_MY_OBJECT(positive_potential_e);
      OT_MY_OBJECT(n_pruned);
      OT_MY_OBJECT(used_error);
    }

   public:

    template<typename TGlobal, typename Tree>
    bool ComputeFiniteDifference(TGlobal &globals,
				 ArrayList<Tree *> &nodes,
				 const Vector &total_n_minus_one_tuples) {

      // If any of the distance evaluation resulted in zero minimum
      // distance, then return false.
      bool flag = globals.kernel_aux.ComputeFiniteDifference
	(globals, nodes, total_n_minus_one_tuples, negative_potential_bound,
	 negative_potential_e, positive_potential_bound, 
	 positive_potential_e, n_pruned, used_error);

      return flag;
    }

    void SetZero() {
      for(index_t i = 0; i < TKernel::order; i++) {
	negative_potential_bound[i].Init(0, 0);
	positive_potential_bound[i].Init(0, 0);
      }
      negative_potential_e.SetZero();
      positive_potential_e.SetZero();
      n_pruned.SetZero();
      used_error.SetZero();
    }

    void Init(const Vector &total_n_minus_one_tuples) {

      negative_potential_bound.Init(TKernel::order);
      negative_potential_e.Init(TKernel::order);
      positive_potential_bound.Init(TKernel::order);
      positive_potential_e.Init(TKernel::order);
      n_pruned.Init(TKernel::order);
      used_error.Init(TKernel::order);

      // Copy the number of pruned tuples...
      n_pruned.CopyValues(total_n_minus_one_tuples);

      // Initializes to zeros...
      SetZero();
    }    
  };

  class MultiTreeQueryPostponed {
    
   public:

    DRange negative_potential_bound;

    double negative_potential_e;

    DRange positive_potential_bound;

    double positive_potential_e;

    double n_pruned;
    
    double used_error;

    void ApplyDelta(const MultiTreeDelta &delta_in, index_t node_index) {

      negative_potential_bound +=
	delta_in.negative_potential_bound[node_index];
      negative_potential_e += delta_in.negative_potential_e[node_index];
      positive_potential_bound +=
	delta_in.positive_potential_bound[nodex_index];
      positive_potential_e += delta_in.positive_potential_e[node_index];
      n_pruned += delta_in.n_pruned[node_index];
      used_error += delta_in.used_error[node_index];
    }

    void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in) {
      negative_potential_bound += postponed_in.negative_potential_bound;
      negative_potential_e += postponed_in.negative_potential_e;
      positive_potential_bound += postponed_in.positive_potential_bound;
      positive_potential_e += postponed_in.positive_potential_e;
      n_pruned += postponed_in.n_pruned;
      used_error += postponed_in.used_error;
    }

    void SetZero() {
      negative_potential_bound.Init(0, 0);
      negative_potential_e = 0;
      positive_potential_bound.Init(0, 0);
      positive_potential_e = 0;
      n_pruned = 0;
      used_error = 0;
    }

    void Init() {

      // Initializes to zeros...
      SetZero();
    }
  };

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
      n_pruned_l += postponed_in.n_pruned_l;
      used_error_u += postponed_in.used_error_u;
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

  class MultiTreeQueryStat {

   public:

    MultiTreeQueryPostponed postponed;
    
    MultiTreeQuerySummary summary;
    
    double priority;
    
    Vector mean;
    
    index_t count;
    
    bool in_strata;

    double num_precomputed_tuples;

    OT_DEF_BASIC(MultiTreeQueryStat) {
      OT_MY_OBJECT(postponed);
      OT_MY_OBJECT(summary);
      OT_MY_OBJECT(priority);
      OT_MY_OBJECT(mean);
      OT_MY_OBJECT(count);
      OT_MY_OBJECT(in_strata);
      OT_MY_OBJECT(num_precomputed_tuples);
    }
    
   public:

    double SumOfPerDimensionVariances
    (const Matrix &dataset, index_t &start, index_t &count) {

      double total_variance = 0;
      for(index_t i = start; i < start + count; i++) {
	const double *point = dataset.GetColumnPtr(i);
	for(index_t d = 0; d < 3; d++) {
	  total_variance += math::Sqr(point[d] - mean[d]);
	}
      }
      total_variance /= ((double) count);
      return total_variance;
    }

    void FinalPush(MultiTreeQueryStat &child_stat) {
      child_stat.postponed.ApplyPostponed(postponed);
    }
    
    void SetZero() {
      postponed.SetZero();
      summary.SetZero();
      priority = 0;
      mean.SetZero();
      in_strata = false;
      num_precomputed_tuples = 0;
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count_in) {
      postponed.Init();
      mean.Init(3);
      SetZero();
      count = count_in;

      // Compute the mean vector.
      for(index_t i = start; i < start + count; i++) {
	const double *point = dataset.GetColumnPtr(i);
	la::AddTo(3, point, mean.ptr());
      }
      la::Scale(3, 1.0 / ((double) count), mean.ptr());

      // Compute the priority of this node which is basically the
      // number of points times sum of per-dimension variances.
      double sum_of_per_dimension_variances = SumOfPerDimensionVariances
	(dataset, start, count);
      priority = count * sum_of_per_dimension_variances;
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count_in,
	      const MultiTreeQueryStat& left_stat, 
	      const MultiTreeQueryStat& right_stat) {
      postponed.Init();
      mean.Init(3);
      SetZero();
      count = count_in;

      la::ScaleOverwrite(left_stat.count, left_stat.mean, &mean);
      la::AddExpert(3, right_stat.count, right_stat.mean.ptr(), mean.ptr());
      la::Scale(3, 1.0 / ((double) count), mean.ptr());

      // Compute the priority of this node which is basically the
      // number of points times sum of per-dimension variances.
      double sum_of_per_dimension_variances = SumOfPerDimensionVariances
	(dataset, start, count);
      priority = count * sum_of_per_dimension_variances;
    }
    
    template<typename TKernelAux>
    void Init(const TKernelAux &kernel_aux_in) {
    }
    
    template<typename TBound, typename TKernelAux>
    void Init(const TBound &bounding_primitive,
	      const TKernelAux &kernel_aux_in) {
      
      // Reset the postponed quantities to zero.
      SetZero();
    }
  };

  class MultiTreeReferenceStat {
  };

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

    OT_DEF_BASIC(MultiTreeQueryResult) {
      OT_MY_OBJECT(negative_potential_bound);
      OT_MY_OBJECT(negative_potential_e);
      OT_MY_OBJECT(positive_potential_bound);
      OT_MY_OBJECT(positive_potential_e);
      OT_MY_OBJECT(final_results);
      OT_MY_OBJECT(n_pruned);
      OT_MY_OBJECT(used_error);
    }

   public:

    void MaximumRelativeError(const MultiTreeQueryResult &other_results,
			      double *max_relative_error,
			      double *negative_max_relative_error,
			      double *positive_max_relative_error) {

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

      la::Add(final_results.length(), negative_potential_e.ptr(),
	      positive_potential_e.ptr(), final_results.ptr());
    }

    template<typename MultiTreeGlobal>
    void Finalize(const MultiTreeGlobal &globals,
		  const ArrayList<index_t> &mapping) {

      MultiTreeUtility::ShuffleAccordingToQueryPermutation
	(final_results, mapping);
    }

    void PrintDebug(const char *output_file_name) const {

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
    }
  };

  /** @brief Defines the global variable for the Axilrod-Teller force
   *         computation.
   */
  class MultiTreeGlobal {

   public:
    
    /** @brief The module holding the parameters.
     */
    struct datanode *module;

    /** @brief The kernel object.
     */
    TKernel kernel_aux;

    /** @brief The chosen indices.
     */
    ArrayList<index_t> hybrid_node_chosen_indices;

    ArrayList<index_t> query_node_chosen_indices;
    
    ArrayList<index_t> reference_node_chosen_indices;

    /** @brief The total number of 3-tuples that contain a particular
     *         particle.
     */
    double total_n_minus_one_tuples;

   public:

    void Init(index_t total_num_particles, index_t dimension_in,
	      const ArrayList<Matrix *> &reference_targets,
	      struct datanode *module_in) {

      kernel_aux.Init();
      hybrid_node_chosen_indices.Init(TKernel::order);
      
      total_n_minus_one_tuples = 
	math::BinomialCoefficient(total_num_particles - 1,
				  TKernel::order - 1);

      // Set the incoming module for referring to parameters.
      module = module_in;
    }

  };

  /** @brief The order of interaction is 3-tuple problem.
   */
  static const int order = TKernel::order;

  static const int num_hybrid_sets = TKernel::order;
  
  static const int num_query_sets = 0;

  static const int num_reference_sets = 0;

  static const double relative_error_ = 0.1;

  template<typename MultiTreeGlobal, typename MultiTreeQueryResult,
	   typename HybridTree, typename QueryTree, typename ReferenceTree>
  static bool ConsiderTupleExact(MultiTreeGlobal &globals,
				 MultiTreeQueryResult &results,
				 MultiTreeDelta &delta,
				 const ArrayList<Matrix *> &query_sets,
				 const ArrayList<Matrix *> &reference_sets,
				 const ArrayList<Matrix *> &reference_targets,
				 ArrayList<HybridTree *> &hybrid_nodes,
				 ArrayList<QueryTree *> &query_nodes,
				 ArrayList<ReferenceTree *> &reference_nodes,
				 double total_num_tuples,
				 double total_n_minus_one_tuples_root,
				 Vector &total_n_minus_one_tuples) {

    // Compute delta change for each node...
    if(hybrid_nodes[0] == hybrid_nodes[1] && 
       hybrid_nodes[1] == hybrid_nodes[2]) {

      for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {
	total_n_minus_one_tuples[i] -= 
	  (hybrid_nodes[i]->stat().num_precomputed_tuples);
      }
    }

    delta.Init(total_n_minus_one_tuples);

    if(unlikely(hybrid_nodes[0]->stat().in_strata && 
		hybrid_nodes[0] == hybrid_nodes[1] &&
		hybrid_nodes[1] == hybrid_nodes[2])) {
      return true;
    }

    if(!delta.ComputeFiniteDifference(globals, hybrid_nodes,
				      total_n_minus_one_tuples)) {
      return false;
    }
    
    // Consider each node in turn whether it can be pruned or not.
    for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {

      // Refine the summary statistics from the new info...
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	AxilrodTellerForceProblem::MultiTreeQuerySummary new_summary;
	new_summary.InitCopy(hybrid_nodes[i]->stat().summary);
	new_summary.ApplyPostponed(hybrid_nodes[i]->stat().postponed);
	new_summary.ApplyDelta(delta, i);

	double sum = new_summary.l1_norm_negative_force_vector_u +
	  new_summary.l1_norm_positive_force_vector_l;
	
	if((AxilrodTellerForceProblem::relative_error_ * sum -
	    (new_summary.used_error_u + 
	     new_summary.probabilistic_used_error_u)) * 
	   total_n_minus_one_tuples[i] < 
	   delta.used_error[i] * 
	   (total_n_minus_one_tuples_root - new_summary.n_pruned_l)) {

	  /*
	  if(((sum + delta.used_error[i]) - sum) >
	     sum * AxilrodTellerForceProblem::relative_error_) {
	    return false;
	  }
	  */

	  const double *negative_force_vector_e = 
	    delta.negative_force_vector_e.GetColumnPtr(i);
	  const double *positive_force_vector_e =
	    delta.positive_force_vector_e.GetColumnPtr(i);
	  double change_l1_norm = 
	    fabs(negative_force_vector_e[0] + negative_force_vector_e[1] +
		 negative_force_vector_e[2] +
		 positive_force_vector_e[0] + positive_force_vector_e[1] +
		 positive_force_vector_e[2]);
	    
	  if(change_l1_norm *
	     (total_n_minus_one_tuples_root - new_summary.n_pruned_l) >
	     sum * AxilrodTellerForceProblem::relative_error_ *
	     total_n_minus_one_tuples[i]) {
	    return false;
	  }
	}
      }
    }

    // In this case, add the delta contributions to the postponed
    // slots of each node.
    for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	hybrid_nodes[i]->stat().postponed.ApplyDelta(delta, i);
      }
    }
    
    results.num_finite_difference_prunes++;
    return true;
  }

  template<typename MultiTreeGlobal, typename MultiTreeQueryResult,
	   typename HybridTree, typename QueryTree, typename ReferenceTree>
  static bool ConsiderTupleProbabilistic
  (MultiTreeGlobal &globals, MultiTreeQueryResult &results,
   MultiTreeDelta &exact_delta, const ArrayList<Matrix *> &query_sets,
   const ArrayList<Matrix *> &sets, ArrayList<HybridTree *> &hybrid_nodes,
   ArrayList<QueryTree *> &query_nodes,
   ArrayList<ReferenceTree *> &reference_nodes,
   double total_num_tuples, double total_n_minus_one_tuples_root,
   const Vector &total_n_minus_one_tuples) {

    // Let's not do Monte Carlo sampling for multibody potentials...
    return false;
  }
    
  static void HybridNodeEvaluateMain(MultiTreeGlobal &globals,
				     const ArrayList<Matrix *> &query_sets,
				     const ArrayList<Matrix *> &sets,
				     const ArrayList<Matrix *> &targets,
				     MultiTreeQueryResult &query_results) {
    
    // The bruteforce case: call the kernel potential.
    globals.kernel_aux.EvaluateMain(globals, sets, query_results);
  }

  static void ReferenceNodeEvaluateMain(MultiTreeGlobal &globals,
					const ArrayList<Matrix *> &query_sets,
					const ArrayList<Matrix *> &sets,
					const ArrayList<Matrix *> &targets,
					MultiTreeQueryResult &query_results) {
    
    // Do nothing because multibody potential is not a query-reference
    // problem...
  }

};

#endif
