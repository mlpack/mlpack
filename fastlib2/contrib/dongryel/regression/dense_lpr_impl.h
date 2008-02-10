// Make sure this file is included only in dense_lpr.h. This is not a
// public header file!
#ifndef INSIDE_DENSE_LPR_H
#error "This file is not a public header file!"
#endif

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::ComputeTargetWeightedReferenceVectors_
(ReferenceTree *rnode) {
  
  if(rnode->is_leaf()) {
    
    // Clear the sum statistics before accumulating.
    (rnode->stat().sum_target_weighted_data_).SetZero();

    // For a leaf reference node, iterate over each reference point
    // and compute the weighted vector and tally these up for the
    // sum statistics owned by the reference node.
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the pointer to the current reference point.
      const double *r_col = rset_.GetColumnPtr(r);

      // Get the pointer to the reference column to be updated.
      double *r_target_weighted_by_coordinates = 
	target_weighted_rset_.GetColumnPtr(r);

      // The first component is the reference target itself.
      r_target_weighted_by_coordinates[0] = rset_targets_[r];

      for(index_t d = 0; d < dimension_; d++) {
	r_target_weighted_by_coordinates[d + 1] = rset_targets_[r] * r_col[d];
      }

      // Tally up the weighted targets.
      la::AddTo(row_length_, r_target_weighted_by_coordinates,
		(rnode->stat().sum_target_weighted_data_).ptr());
    }
    
    // Compute L1 norm of the accumulated sum
    rnode->stat().l1_norm_sum_target_weighted_data_ =
      MatrixUtil::L1Norm(rnode->stat().sum_target_weighted_data_);
  }  
  else {
    
    // Recursively call the function with left and right and merge.
    ComputeTargetWeightedReferenceVectors_(rnode->left());
    ComputeTargetWeightedReferenceVectors_(rnode->right());
    
    la::AddOverwrite((rnode->left()->stat()).sum_target_weighted_data_,
		     (rnode->right()->stat()).sum_target_weighted_data_,
		     &(rnode->stat().sum_target_weighted_data_));
    rnode->stat().l1_norm_sum_target_weighted_data_ =
      MatrixUtil::L1Norm(rnode->stat().sum_target_weighted_data_);
  }
}
