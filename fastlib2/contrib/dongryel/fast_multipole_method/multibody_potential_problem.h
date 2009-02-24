#ifndef MULTIBODY_POTENTIAL_PROBLEM_H
#define MULTIBODY_POTENTIAL_PROBLEM_H

#include "fastlib/fastlib.h"

#include "mlpack/series_expansion/kernel_aux.h"
#include "mlpack/kde/inverse_normal_cdf.h"

#include "at_potential_kernel.h"
#include "three_body_gaussian_kernel.h"
#include "../multitree_template/multitree_utility.h"

template<typename TKernel>
class MultibodyPotentialProblem {

 public:

  #include "mbp_delta.h"

  #include "mbp_query_postponed.h"

  #include "mbp_query_summary.h"

  #include "mbp_stat.h"

  #include "mbp_query_result.h"

  #include "mbp_global.h"

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

    // If the monochromatic trick was applied, then readjust the
    // number of remaining tuples...
    if(hybrid_nodes[0] == hybrid_nodes[1] && 
       hybrid_nodes[1] == hybrid_nodes[2]) {

      for(index_t i = 0; i < TKernel::order; i++) {
	total_n_minus_one_tuples[i] -= 
	  (hybrid_nodes[i]->stat().num_precomputed_tuples);
      }
    }

    // Compute delta change for each node...
    delta.Init(total_n_minus_one_tuples);

    // Compute delta change for each node...
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
    for(index_t i = 0; i < TKernel::order; i++) {

      // Refine the summary statistics from the new info...
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	typename MultibodyPotentialProblem<TKernel>::MultiTreeQuerySummary 
	  new_summary;
	new_summary.InitCopy(hybrid_nodes[i]->stat().summary);
	new_summary.ApplyPostponed(hybrid_nodes[i]->stat().postponed);
	new_summary.ApplyDelta(delta, i);

	// Compute the right hand side of the pruning rule
	double right_hand_side = 
	  (globals.relative_error *
	   (new_summary.positive_potential_bound.lo -
	    new_summary.negative_potential_bound.hi) -
	   new_summary.used_error_u) * 
	  (total_n_minus_one_tuples[i] / 
	   (total_n_minus_one_tuples_root - new_summary.n_pruned_l));

	if(delta.used_error[i] > right_hand_side) {
	  return false;
	}
      }
    }
    
    // In this case, add the delta contributions to the postponed
    // slots of each node.
    for(index_t i = 0; i < TKernel::order; i++) {
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

    // Perhaps, this should be parameterized by the Global object.
    const int num_samples = 100;

    // If Monte Carlo approximation is disabled or the three nodes
    // contain too few number of points, then return.
    index_t max_number_of_points = 0;
    bool overlap = false;
    for(index_t i = 0; i < hybrid_nodes.size(); i++) {
      max_number_of_points = std::max(max_number_of_points, 
				      hybrid_nodes[i]->count());
      if(i > 0) {
	overlap = (hybrid_nodes[i] == hybrid_nodes[i - 1]);
      }
      if(overlap) {
	break;
      }
    }

    if(globals.probability >= 1 || max_number_of_points < num_samples ||
       overlap) {
      return false;
    }

    // Sample the n-tuples and compute the sample mean and the
    // variance.
    double potential_sums = 0;
    double potential_avg = 0;
    double standard_deviation = 0;
    double squared_potential_sums = 0;

    for(index_t i = 0; i < num_samples; i++) {

      // A for-loop and a do-while to choose a unique n-tuple.
      for(index_t j = 0; j < TKernel::order; j++) {

	do {
	  globals.hybrid_node_chosen_indices[j] = 
	    math::RandInt(hybrid_nodes[j]->begin(), hybrid_nodes[j]->end());

	} while(j != 0 && globals.hybrid_node_chosen_indices[j] ==
		globals.hybrid_node_chosen_indices[j - 1]);

      } // end of the for-loop for choosing the indices...

      // Compute the potential value for the chosen indices.
      double potential = globals.kernel_aux.EvaluateMain(globals, sets);      
      potential_sums += potential;
      squared_potential_sums += math::Sqr(potential);

    } // end of iterating over each sample...

    potential_avg = potential_sums / ((double) num_samples);
    standard_deviation = 
      sqrt((squared_potential_sums - num_samples * math::Sqr(potential_avg))
	   / ((double) num_samples - 1)) / sqrt(num_samples);

    // Refine delta bounds based on sampling.
    exact_delta.RefineBounds(globals, potential_avg, standard_deviation);

    // Consider each node in turn whether it can be pruned or not.
    for(index_t i = 0; i < TKernel::order; i++) {

      // Refine the summary statistics from the new info...
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	typename MultibodyPotentialProblem<TKernel>::MultiTreeQuerySummary 
	  new_summary;
	new_summary.InitCopy(hybrid_nodes[i]->stat().summary);
	new_summary.ApplyPostponed(hybrid_nodes[i]->stat().postponed);
	new_summary.ApplyDelta(exact_delta, i);

	// Compute the right hand side of the pruning rule.
	double right_hand_side = 
	  (globals.relative_error *
	   (new_summary.positive_potential_bound.lo -
	    new_summary.negative_potential_bound.hi) -
	   new_summary.used_error_u) * 
	  (total_n_minus_one_tuples[i] / 
	   (total_n_minus_one_tuples_root - new_summary.n_pruned_l));

	if(exact_delta.used_error[i] > right_hand_side) {
	  return false;
	}
      }
    }
    
    // In this case, add the delta contributions to the postponed
    // slots of each node.
    for(index_t i = 0; i < TKernel::order; i++) {
      if(i == 0 || hybrid_nodes[i] != hybrid_nodes[i - 1]) {
	hybrid_nodes[i]->stat().postponed.ApplyDelta(exact_delta, i);
      }
    }
    
    results.num_monte_carlo_prunes++;
    return true;
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
