// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void LocalLinearKrylov<TKernel>::InitializeQueryTreeRightHandSides_
(Tree *qnode) {
  
  // Set the bounds to default values.
  (qnode->stat().right_hand_sides_l_).SetZero();
  (qnode->stat().right_hand_sides_u_).CopyValues
    (rroot_->stat().sum_targets_weighted_by_data_);
  (qnode->stat().postponed_right_hand_sides_l_).SetZero();
  (qnode->stat().postponed_right_hand_sides_e_).SetZero();
  (qnode->stat().postponed_right_hand_sides_u_).SetZero();

  // If the query node is not a leaf node, then traverse to the left
  // and the right.
  if(!qnode->is_leaf()) {
    InitializeQueryTreeRightHandSides_(qnode->left());
    InitializeQueryTreeRightHandSides_(qnode->right());
  }
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::ComputeWeightedTargetVectors_(Tree *rnode) {
  
  if(rnode->is_leaf()) {
    
    // Clear the sum statistics before accumulating.
    (rnode->stat().sum_targets_weighted_by_data_).SetZero();

    // For a leaf reference node, iterate over each reference point
    // and compute the weighted vector and tally these up for the
    // sum statistics owned by the reference node.
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the pointer to the current reference point.
      const double *r_col = rset_.GetColumnPtr(r);

      // Get the pointer to the reference column to be updated.
      double *r_target_weighted_by_coordinates = 
	rset_targets_weighted_by_coordinates_.GetColumnPtr(r);

      // The first component is the reference target itself.
      r_target_weighted_by_coordinates[0] = rset_targets_[r];

      for(index_t d = 0; d < dimension_; d++) {
	r_target_weighted_by_coordinates[d + 1] = rset_targets_[r] * r_col[d];
      }
      la::AddTo(row_length_, r_target_weighted_by_coordinates,
		(rnode->stat().sum_targets_weighted_by_data_).ptr());
    }
    
    // Compute L1 norm of the accumulated sum
    rnode->stat().l1_norm_sum_targets_weighted_by_data_ =
      L1Norm_(rnode->stat().sum_targets_weighted_by_data_);
  }
  else {
    
    // Recursively call the function with left and right and merge.
    ComputeWeightedTargetVectors_(rnode->left());
    ComputeWeightedTargetVectors_(rnode->right());
    
    la::AddOverwrite((rnode->left()->stat()).sum_targets_weighted_by_data_,
		     (rnode->right()->stat()).sum_targets_weighted_by_data_,
		     &(rnode->stat().sum_targets_weighted_by_data_));
    rnode->stat().l1_norm_sum_targets_weighted_by_data_ =
      (rnode->left()->stat()).l1_norm_sum_targets_weighted_by_data_ +
      (rnode->right()->stat()).l1_norm_sum_targets_weighted_by_data_;
  }
}
