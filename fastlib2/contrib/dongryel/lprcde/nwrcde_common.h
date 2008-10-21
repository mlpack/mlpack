#ifndef NWRCDE_COMMON_H
#define NWRCDE_COMMON_H

#include "fastlib/fastlib.h"

class NWRCdeCommon {

 public:

  /** @brief Shuffles a vector according to a given permutation.
   *
   *  @param v The vector to be shuffled.
   *  @param permutation The permutation.
   */
  static void ShuffleAccordingToPermutation
  (Vector &v, const ArrayList<index_t> &permutation) {
    
    Vector v_tmp;
    v_tmp.Init(v.length());
    for(index_t i = 0; i < v_tmp.length(); i++) {
      v_tmp[i] = v[permutation[i]];
    }
    v.CopyValues(v_tmp);
  }

  /** @brief Shuffles a vector according to a given permutation.
   *
   *  @param v The vector to be shuffled.
   *  @param permutation The permutation.
   */
  static void ShuffleAccordingToPermutation
  (Matrix &v, const ArrayList<index_t> &permutation) {
    
    for(index_t i = 0; i < v.n_cols(); i++) {
      Vector column_vector;
      v.MakeColumnVector(i, &column_vector);
      NWRCdeCommon::ShuffleAccordingToPermutation(column_vector, permutation);
    }
  }

  template<typename Tree1, typename Tree2>
  static void Heuristic
  (Tree1 *nd, Tree2 *nd1, Tree2 *nd2, double probability, 
   Tree2 **partner1, double *probability1, 
   Tree2 **partner2, double *probability2) {
    
    double d1 = nd->bound().MinDistanceSq(nd1->bound());
    double d2 = nd->bound().MinDistanceSq(nd2->bound());
    
    // Prioritized traversal based on the squared distance bounds.
    if(d1 <= d2) {
      *partner1 = nd1;
      *probability1 = sqrt(probability);
      *partner2 = nd2;
      *probability2 = sqrt(probability);
    }
    else {
      *partner1 = nd2;
      *probability1 = sqrt(probability);
      *partner2 = nd1;
      *probability2 = sqrt(probability);
    }
  }

  template<typename TGlobal, typename QueryTree, typename ReferenceTree,
	   typename TQueryResult, typename TDelta>
  static void ApplySeriesExpansion(const TGlobal &globals, const Matrix &qset,
				   QueryTree *qnode, ReferenceTree *rnode, 
				   TQueryResult &query_results,
				   const TDelta &delta) {

    // FIX ME for multi-target NWR code!
    Vector nwr_numerator_weights_alias;
    globals.nwr_numerator_weights.MakeColumnVector
      (0, &nwr_numerator_weights_alias);

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
	  (globals.rset, nwr_numerator_weights_alias, rnode->begin(), 
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
	  (globals.rset, globals.nwr_denominator_weights, rnode->begin(),
	   rnode->end(), delta.nwr_denominator.order_local);
	query_results.num_direct_local_prunes++;
	break;
      default:
	break;
    }

    // Lastly, we need to add on the rest of the delta contributions.
    qnode->stat().postponed.ApplyDelta(delta);
  }

  template<typename TGlobal, typename QueryTree, typename ReferenceTree,
	   typename TQueryResult, typename TQuerySummary, typename TDelta,
	   typename TError>
  static bool ConsiderPairExact
  (const TGlobal &parameters, const Matrix &qset, QueryTree *qnode,
   ReferenceTree *rnode, double probability, TQueryResult &query_results,
   TQuerySummary &new_summary, TDelta &delta, TError &allowed_error) {
    
    // Compute the bound changes due to a finite-difference approximation.
    delta.ComputeFiniteDifference(parameters, qnode, rnode);

    // Refine the lower bound using the new lower bound info.
    new_summary.InitCopy(qnode->stat().summary);
    new_summary.ApplyPostponed(qnode->stat().postponed);
    new_summary.ApplyDelta(delta);
    
    // Compute the allowable error.
    allowed_error.ComputeAllowableError(parameters, new_summary, rnode);

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
      else if(delta.ComputeSeriesExpansion(parameters, qnode, rnode, 
					   allowed_error)) {
	ApplySeriesExpansion(parameters, qset, qnode, rnode, query_results,
			     delta);
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

};

#endif
