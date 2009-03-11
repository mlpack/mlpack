#ifndef KDE_PROBLEM_H
#define KDE_PROBLEM_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "../multitree_template/multitree_utility.h"
#include "mlpack/kde/inverse_normal_cdf.h"

#define INSIDE_KDE_PROBLEM_H

template<typename TKernelAux>
class KdeProblem {

 public:

  static const int num_hybrid_sets = 0;

  static const int num_query_sets = 1;

  static const int num_reference_sets = 1;

  static const int order = 2;

  // Include the MultiTreeDelta class definition.
  #include "kde_delta.h"

  // Include the Error class for storing the approximation error.
  #include "kde_error.h"

  // Include the MultiTreeGlobal class definition.
  #include "kde_global.h"

  // Include the MultiTreeQueryResult class definition.
  #include "kde_query_result.h"

  // Include the MultiTreeQueryPostponed class definition.
  #include "kde_query_postponed.h"

  // Include the MultiTreeQuerySummary class definition.
  #include "kde_query_summary.h"

  // Include the MultiTreeQueryStat and MultiTreeReferenceStat class
  // definitions.
  #include "kde_stat.h"
  
  template<typename TGlobal, typename QueryTree, typename ReferenceTree,
	   typename TQueryResult, typename TDelta>
  static void ApplySeriesExpansion(const TGlobal &globals, const Matrix &qset,
				   const Matrix &rset,
				   const Matrix &reference_weights,
				   QueryTree *qnode, ReferenceTree *rnode, 
				   TQueryResult &query_results,
				   const TDelta &delta) {

    Vector nwr_numerator_weights_alias;
    reference_weights.MakeColumnVector(0, &nwr_numerator_weights_alias);

    switch(delta.kde_approximation.approx_type) {
      case TDelta::FAR_TO_LOCAL:
	DEBUG_ASSERT(delta.kde_approximation.order_farfield_to_local >= 0);
	rnode->stat().farfield_expansion.TranslateToLocal
	  (qnode->stat().local_expansion, 
	   delta.kde_approximation.order_farfield_to_local);
	query_results.num_far_to_local_prunes++;
	break;
      case TDelta::DIRECT_FARFIELD:
	DEBUG_ASSERT(delta.kde_approximation.order_farfield >= 0);
	for(index_t q = qnode->begin(); q < qnode->end(); q++) {
	  query_results.sum_e[q] += 
	    rnode->stat().farfield_expansion.EvaluateField
	    (qset, q, delta.kde_approximation.order_farfield);
	}
	query_results.num_direct_far_prunes++;
	break;
      case TDelta::DIRECT_LOCAL:
	DEBUG_ASSERT(delta.kde_approximation.order_local >= 0);
	qnode->stat().local_expansion.AccumulateCoeffs
	  (rset, nwr_numerator_weights_alias, rnode->begin(), 
	   rnode->end(), delta.kde_approximation.order_local);
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
    typename KdeProblem::KdeError allowed_error;
    allowed_error.ComputeAllowableError(globals, new_summary, delta, rnode);

    // If the error bound is satisfied by the hard error bound, it is
    // safe to prune.
    if(!isnan(allowed_error.error)) {
      
      if(delta.kde_approximation.used_error <= allowed_error.error) {

	qnode->stat().postponed.ApplyDelta(delta);
	query_results.num_finite_difference_prunes++; 
	return true;
      }

      // Try series expansion.
      else if(globals.dimension <= 6 &&
	      delta.ComputeSeriesExpansion(globals, qnode, rnode, 
					   allowed_error)) {
	KdeProblem::ApplySeriesExpansion(globals, *(query_sets[0]),
					 *(reference_sets[0]),
					 *(targets[0]), qnode,
					 rnode, query_results, delta);
	return true;
      }

      else {
	
	if(qnode->is_leaf() && rnode->is_leaf()) {
	  
	  const Matrix &query_set = *(query_sets[0]);
	  qnode->stat().summary.StartReaccumulate();
	  
	  for(index_t q = qnode->begin(); q < qnode->end(); q++) {

	    query_results.ApplyPostponed(qnode->stat().postponed, q);
	    const double *query_point = query_set.GetColumnPtr(q);
	    delta.ComputeFiniteDifference(globals, query_point, rnode);
	    new_summary.Init(query_results, q);
	    new_summary.ApplyDelta(delta);
	    allowed_error.ComputeAllowableError(globals, new_summary, delta,
						rnode);

	    if(delta.kde_approximation.used_error > allowed_error.error) {
	      for(index_t r = rnode->begin(); r < rnode->end(); r++) {

		double squared_distance =
		  la::DistanceSqEuclidean(globals.dimension,
					  query_sets[0]->GetColumnPtr(q),
					  reference_sets[0]->GetColumnPtr(r));
		
		double kernel_value = globals.kernel_aux.kernel_.EvalUnnormOnSq
		  (squared_distance);

		query_results.sum_l[q] += kernel_value;
		query_results.sum_e[q] += kernel_value;

	      } // end of looping over each reference point...

	      query_results.UpdatePrunedComponents(reference_nodes, q);
	    }
	    else {
	      query_results.ApplyDelta(delta, q);
	    }

	    qnode->stat().summary.Accumulate(query_results, q);
	    
	  } // end of looping over each query point

	  qnode->stat().postponed.SetZero();
	  return true;

	}   // end of the case for both query and reference being leaves...
	else {
	  return false;
	}
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
   MultiTreeDelta &exact_delta, const ArrayList<Matrix *> &query_sets,
   const ArrayList<Matrix *> &sets, ArrayList<HybridTree *> &nodes,
   ArrayList<QueryTree *> &query_nodes,
   ArrayList<ReferenceTree *> &reference_nodes, double total_num_tuples,
   double total_n_minus_one_tuples_root,
   const Vector &total_n_minus_one_tuples) {

    if(globals.probability >= 1) {
      return false;
    }

    QueryTree *qnode = query_nodes[0];
    ReferenceTree *rnode = reference_nodes[0];
    const Matrix &qset = *(query_sets[0]);
    const Matrix &rset = *(sets[0]);

    // If the reference node contains too few points, then return.
    if(qnode->count() * rnode->count() < 50) {
      return false;
    }

    // Refine the lower bound using the new lower bound info.
    MultiTreeQuerySummary new_summary;
    new_summary.InitCopy(qnode->stat().summary);
    new_summary.ApplyPostponed(qnode->stat().postponed);

    // Refine the lower bound using the new lower bound info.
    double max_used_error = 0;

    // Take random query/reference pair samples and determine how many
    // more samples are needed.
    bool flag = true;

    // Temporary sum accumulants.
    double kernel_sums = 0;
    double kernel_sums_l = 0;
    double squared_kernel_sums = 0;

    // Commence sampling...
    {
      double standard_score =
        InverseNormalCDF::Compute(globals.probability + 0.5 * 
				  (1 - globals.probability));

      // The initial number of samples is equal to the default.
      int num_samples = 25;
      int total_samples = 0;

      for(index_t s = 0; s < num_samples; s++) {
	
	index_t random_query_point_index =
	  math::RandInt(qnode->begin(), qnode->end());
	index_t random_reference_point_index =
	  math::RandInt(rnode->begin(), rnode->end());
	
	// Get the pointer to the current query point.
	const double *query_point =
	  qset.GetColumnPtr(random_query_point_index);
	
	// Get the pointer to the current reference point.
	const double *reference_point =
	  rset.GetColumnPtr(random_reference_point_index);
	
	// Compute the pairwise distance and kernel value.
	double squared_distance = la::DistanceSqEuclidean
	  (rset.n_rows(), query_point, reference_point);
	
	double weighted_kernel_value =
	  globals.kernel_aux.kernel_.EvalUnnormOnSq(squared_distance);
	kernel_sums += weighted_kernel_value;
	squared_kernel_sums += weighted_kernel_value * weighted_kernel_value;
	
      } // end of taking samples for this roune...
      
      // Increment total number of samples.
      total_samples += num_samples;
      
      // Compute the current estimate of the sample mean and the
      // sample variance.
      double sample_mean = kernel_sums / ((double) total_samples);
      double sample_variance =
	(squared_kernel_sums - total_samples * sample_mean * sample_mean) /
	((double) total_samples - 1);
      
      // The currently proven lower bound.
      double current_lower_bound = new_summary.sum_l +
	std::max(exact_delta.kde_approximation.sum_l, 
		 rnode->stat().get_weight_sum() * 
		 (sample_mean - sqrt(sample_variance) * standard_score));

      // Right hand side due to the estimates...
      double right_hand_side =
	(globals.relative_error * current_lower_bound -
	 (new_summary.used_error_u + 
	  new_summary.probabilistic_used_error_u)) /
	(globals.num_reference_points - new_summary.n_pruned_l) *
	rnode->stat().get_weight_sum();
      
      if((new_summary.sum_l + rnode->stat().get_weight_sum() * 
	  sqrt(sample_variance) * standard_score) - new_summary.sum_l <= 
	 right_hand_side) {
	
	kernel_sums = sample_mean * rnode->stat().get_weight_sum();
	max_used_error = rnode->stat().get_weight_sum() *
	  standard_score * sqrt(sample_variance);
	kernel_sums_l = kernel_sums - max_used_error;
      }
      else {
	flag = false;
      }

    } // end of sampling...

    // If all queries can be pruned, then add the approximations.
    if(flag) {
      qnode->stat().postponed.sum_l += 
	std::max(kernel_sums_l, exact_delta.kde_approximation.sum_l);
      qnode->stat().postponed.sum_e += kernel_sums;
      qnode->stat().postponed.n_pruned += rnode->count();
      qnode->stat().postponed.probabilistic_used_error =
	sqrt(math::Sqr(qnode->stat().postponed.probabilistic_used_error) +
	     math::Sqr(max_used_error));
      return true;
    }
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

    query_results.sum_l[q_index] += kernel_value;
    query_results.sum_e[q_index] += kernel_value;

  }
};

#undef INSIDE_KDE_PROBLEM_H

#endif
