#ifndef INSIDE_DUALTREE_KDE_CV_H
#error "This is not a public header file!"
#endif

#include "inverse_normal_cdf.h"

template<typename TKernelAux>
void DualtreeKdeCV<TKernelAux>::DualtreeKdeCVBase_(Tree *qnode, Tree *rnode,
						   double probability) {

  // Compute unnormalized sum for each query point.
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {

    // Get the query point.
    const double *q_col = rset_.GetColumnPtr(q);
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // Pairwise distance and kernel values.
      double dsqd = la::DistanceSqEuclidean(rset_.n_rows(), q_col, r_col);
      double first_kernel_value = first_ka_.kernel_.EvalUnnormOnSq(dsqd);
      double first_weighted_kernel_value = rset_weights_[r] * 
	first_kernel_value;
      double second_kernel_value = second_ka_.kernel_.EvalUnnormOnSq(dsqd);
      double second_weighted_kernel_value = rset_weights_[r] *
	second_kernel_value;

      // Accumulate the sum from the computed kernel values.
      first_sum_l_ += first_weighted_kernel_value;
      first_sum_e_ += first_weighted_kernel_value;
      first_sum_u_ += first_weighted_kernel_value;
      second_sum_l_ += second_weighted_kernel_value;
      second_sum_e_ += second_weighted_kernel_value;
      second_sum_u_ += second_weighted_kernel_value;

    } // end of iterating over each reference point.

  } // end of looping over each query point.

  // Increment the number of pruned portions.
  n_pruned_ += (((double) qnode->count()) / ((double) rroot_->count())) * 
    (rnode->stat().get_weight_sum() / rset_weight_sum_);

  // Undo upper bound contributions.
  first_sum_u_ -= qnode->count() * rnode->stat().get_weight_sum();
  second_sum_u_ -= qnode->count() * rnode->stat().get_weight_sum();
}

template<typename TKernelAux>
double DualtreeKdeCV<TKernelAux>::EvalUnnormOnSq_
(index_t reference_point_index,	double squared_distance) {
  return first_ka_.kernel_.EvalUnnormOnSq(squared_distance);
}

template<typename TKernelAux>
bool DualtreeKdeCV<TKernelAux>::DualtreeKdeCVCanonical_
(Tree *qnode, Tree *rnode, double probability) {
  
  // Temporary variables for storing bound changes.
  double first_dl, first_de, first_du, first_used_error, n_pruned;
  double second_dl, second_de, second_du, second_used_error;

  // Temporary variable for holding distance/kernel value bounds.
  DRange dsqd_range, first_kernel_value_range, second_kernel_value_range;
  
  // First compute distance/kernel value bounds.
  dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
  dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
  first_kernel_value_range = first_ka_.kernel_.RangeUnnormOnSq(dsqd_range);
  second_kernel_value_range = second_ka_.kernel_.RangeUnnormOnSq(dsqd_range);

  // Try finite difference pruning first.
  if(DualtreeKdeCVCommon::Prunable
     (qnode, rnode, probability, dsqd_range, first_kernel_value_range, 
      second_kernel_value_range, first_dl, first_de, first_du, 
      first_used_error, second_dl, second_de, second_du, second_used_error,
      n_pruned, this)) {
    first_sum_l_ += first_dl;
    first_sum_e_ += first_de;
    first_sum_u_ += first_du;
    first_used_error_ += first_used_error;
    second_sum_l_ += second_dl;
    second_sum_e_ += second_de;
    second_sum_u_ += second_du;
    second_used_error_ += second_used_error;
    n_pruned_ += n_pruned;
    num_finite_difference_prunes_++;
    return true;
  }
  
  // For a leaf query node,
  if(qnode->is_leaf()) {
    
    // For leaf pairs, do exhaustive computations.
    if(rnode->is_leaf()) {
      DualtreeKdeCVBase_(qnode, rnode, probability);
      return true;
    }
    
    // For a non-leaf reference, expand reference node,
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
      DualtreeKdeCommon::BestNodePartners
	(qnode, rnode->left(), rnode->right(), probability,
	 &rnode_first, &probability_first, &rnode_second, &probability_second);
      
      bool first_result = 
	DualtreeKdeCVCanonical_(qnode, rnode_first, probability_first);

      // If the first recursion is computed exactly, then increment
      // the probability tolerance for the second recursion.
      if(first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool second_result =
	DualtreeKdeCVCanonical_(qnode, rnode_second, probability_second);
      return first_result && second_result;
    }
  }
  
  // For a non-leaf query node,
  else {

    // The boolean flag that states that the contribution of the
    // current reference node is computed exactly for the current
    // query node.
    bool result = true;
    
    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      Tree *qnode_first = NULL, *qnode_second = NULL;
      double probability_first = 0, probability_second = 0;

      DualtreeKdeCommon::BestNodePartners
	(rnode, qnode->left(), qnode->right(), probability,
	 &qnode_first, &probability_first, &qnode_second, &probability_second);
      bool first_result =
	DualtreeKdeCVCanonical_(qnode_first, rnode, probability);
      bool second_result =
	DualtreeKdeCVCanonical_(qnode_second, rnode, probability);
      result = first_result && second_result;
    }
    
    // For a non-leaf reference node, expand both query and reference
    // nodes.
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      double probability_first = 0, probability_second = 0;
  
      // Fix the query node to be the left child, and recurse.
      DualtreeKdeCommon::BestNodePartners
	(qnode->left(), rnode->left(), rnode->right(), probability, 
	 &rnode_first, &probability_first, &rnode_second, &probability_second);
      bool left_first_result =
	DualtreeKdeCVCanonical_(qnode->left(), rnode_first, probability_first);

      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(left_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool left_second_result = 
	DualtreeKdeCVCanonical_(qnode->left(), rnode_second, 
				probability_second);
      
      // Fix the query node to be the right child, and recurse.
      DualtreeKdeCommon::BestNodePartners
	(qnode->right(), rnode->left(), rnode->right(), probability, 
	 &rnode_first, &probability_first, &rnode_second, &probability_second);
      bool right_first_result =
	DualtreeKdeCVCanonical_(qnode->right(), rnode_first, 
				probability_first);

      // If the first recursion is carried out exactly, then increment
      // the probability tolerance for the second recursion.
      if(right_first_result) {
	probability_second = math::Sqr(probability_first);
      }

      bool right_second_result =
	DualtreeKdeCVCanonical_(qnode->right(), rnode_second,
				probability_second);

      result = left_first_result && left_second_result &&
	right_first_result && right_second_result;
    }
    return result;
  } // end of the case: non-leaf query node.

} // end of DualtreeKdeCVCanonical_

template<typename TKernelAux>
void DualtreeKdeCV<TKernelAux>::PreProcess(Tree *node) {

  // Initialize the center of expansions and bandwidth for series
  // expansion.
  Vector bounding_box_center;
  node->stat().Init(first_ka_, second_ka_);
  node->bound().CalculateMidpoint(&bounding_box_center);
  (node->stat().first_farfield_expansion_.get_center())->CopyValues
    (bounding_box_center);
  (node->stat().second_farfield_expansion_.get_center())->CopyValues
    (bounding_box_center);
  
  // for non-leaf node, recurse
  if(!node->is_leaf()) {
    
    PreProcess(node->left());
    PreProcess(node->right());
    
    // Translate the multipole moments.
    node->stat().first_farfield_expansion_.TranslateFromFarField
      (node->left()->stat().first_farfield_expansion_);
    node->stat().first_farfield_expansion_.TranslateFromFarField
      (node->right()->stat().first_farfield_expansion_);
    node->stat().second_farfield_expansion_.TranslateFromFarField
      (node->left()->stat().second_farfield_expansion_);
    node->stat().second_farfield_expansion_.TranslateFromFarField
      (node->right()->stat().second_farfield_expansion_);      
  }
  else {
    
    // Exhaustively compute multipole moments.
    node->stat().first_farfield_expansion_.RefineCoeffs
      (rset_, rset_weights_, node->begin(), node->end(),
       first_ka_.sea_.get_max_order());
    node->stat().second_farfield_expansion_.RefineCoeffs
      (rset_, rset_weights_, node->begin(), node->end(),
       first_ka_.sea_.get_max_order());    
  }
}
