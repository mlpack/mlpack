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

      // Compute the multiindex expansion of the given reference point.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order, r_col, r_target_weighted_by_coordinates);
      
      // Scale the expansion by the reference target.
      la::Scale(row_length_, rset_targets_[r], 
		r_target_weighted_by_coordinates);
      
      // Tally up the weighted targets.
      la::AddTo(row_length_, r_target_weighted_by_coordinates,
		(rnode->stat().sum_target_weighted_data_).ptr());
    }
    
    // Compute Frobenius norm of the accumulated sum
    rnode->stat().frobenius_norm_sum_target_weighted_data_ =
      MatrixUtil::FrobeniusNorm(rnode->stat().sum_target_weighted_data_);
  }  
  else {
    
    // Recursively call the function with left and right and merge.
    ComputeTargetWeightedReferenceVectors_(rnode->left());
    ComputeTargetWeightedReferenceVectors_(rnode->right());
    
    la::AddOverwrite((rnode->left()->stat()).sum_target_weighted_data_,
		     (rnode->right()->stat()).sum_target_weighted_data_,
		     &(rnode->stat().sum_target_weighted_data_));
    rnode->stat().frobenius_norm_sum_target_weighted_data_ =
      rnode->left()->stat().frobenius_norm_sum_target_weighted_data_ +
      rnode->right()->stat().frobenius_norm_sum_target_weighted_data_;
  }
}

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::DualtreeLprBase_
(QueryTree *qnode, ReferenceTree *rnode) {

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().numerator_l_.SetAll(DBL_MAX);
  qnode->stat().numerator_u_.SetAll(-DBL_MAX);
  qnode->stat().numerator_used_error_ = 0;
  qnode->stat().numerator_n_pruned_ = DBL_MAX;
  qnode->stat().denominator_l_.SetAll(DBL_MAX);
  qnode->stat().denominator_u_.SetAll(-DBL_MAX);
  qnode->stat().denominator_used_error_ = 0;
  qnode->stat().denominator_n_pruned_ = DBL_MAX;
  
  // compute unnormalized sum
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // Temporary variable for storing multivariate expansion of a
    // reference point.
    Vector reference_point_expansion;
    reference_point_expansion.Init(row_length_);

    // Incorporate the postponed information for the numerator matrix.
    la::AddTo(row_length_, qnode->stat().postponed_numerator_l_.ptr(), 
	      numerator_l_.GetColumnPtr(q));
    la::AddTo(row_length_, qnode->stat().postponed_numerator_u_, 
	      numerator_u_.GetColumnPtr(q));
    numerator_used_error_[q] += qnode->stat().postponed_numerator_used_error_;
    numerator_n_pruned_[q] += qnode->stat().postponed_n_pruned_;

    // Incorporate the postponed information for the denominator matrix.
    la::AddTo(qnode->stat().postponed_denominator_l_, denominator_l_[q]);
    la::AddTo(qnode->stat().postponed_denominator_u_, denominator_u_[q]);
    denominator_used_error_[q] += 
      qnode->stat().postponed_denominator_used_error_;
    denominator_n_pruned_[q] += qnode->stat().postponed_denominator_n_pruned_;
    
    // Get the query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // Get the query-relevant quantities to be updated.
    double *q_numerator_l = numerator_l_.GetColumnPtr(q);
    double *q_numerator_e = numerator_e_.GetColumnPtr(q);
    double *q_numerator_u = numerator_u_.GetColumnPtr(q);
    Matrix &q_denominator_l = denominator_l_[q];
    Matrix &q_denominator_e = denominator_e_[q];
    Matrix &q_denominator_u = denominator_u_[q];

    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point and its training value.
      const double *r_col = rset_.GetColumnPtr(r);

      // Compute the reference point expansion.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order, r_col, reference_point_expansion.ptr());

      // Pairwise distance and kernel value and kernel value weighted
      // by the reference target training value.
      double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
      double kernel_value = kernel_.EvalUnnormOnSq(dsqd);
      double target_weighted_kernel_value = rset_targets_[r] * kernel_value;
      
      // Loop over each column of the matrix to be updated.
      for(index_t j = 0; j < row_length_; j++) {

	// Tally the sum up for the numerator vector B^T W(q) Y.
	q_numerator_l[j] += target_weighted_kernel_value * 
	  reference_point_expansion[j];
	q_numerator_e[j] += target_weighted_kernel_value *
	  reference_point_expansion[j];
	q_numerator_u[j] += target_weighted_kernel_value * 
	  reference_point_expansion[j];
	
	// Loop over each row of the matrix to be updated.
	for(index_t i = 0; i < row_length_; i++) {
	  
	  // Tally the sum up for the denominator matrix B^T W(q) B.
	  q_denominator_l.set(i, j, q_denominator_l.get(i, j) +
			      kernel_value * 
			      reference_point_expansion[i] *
			      reference_point_expansion[j]);
	} // End of iterating over each row.
      } // End of iterating over each column.

    } // End of iterating over each reference point.
    
    // each query point has taken care of all reference points.
    numerator_n_pruned_[q] += 
      rnode->stat().frobenius_norm_sum_target_weighted_data_;
    denominator_n_pruned_[q] +=
      rnode->stat().frobenius_norm_sum_data_outer_products_;
    
    // Subtract from the upper bounds to undo the assumption made in
    // the preprocessing function.
    la::SubFrom(row_length_, rnode->stat().sum_target_weighted_data_.ptr(), 
		numerator_u_.GetColumnPtr(q));
    la::SubFrom(rnode->stat().sum_data_outer_products_, &(denominator_u_[q]));
    
    // Refine min and max summary statistics for the numerator.
    MatrixUtil::ComponentwiseMin(row_length_,
				 qnode->stat().numerator_l_.GetColumnPtr(q),
				 numerator_l_.GetColumnPtr(q),
				 qnode->stat().numerator_l_.GetColumnPtr(q));
    MatrixUtil::ComponentwiseMax(row_length_,
				 qnode->stat().numerator_u_.GetColumnPtr(q),
				 numerator_u_.GetColumnPtr(q),
				 qnode->stat().numerator_u_.GetColumnPtr(q));
    qnode->stat().numerator_used_error_ =
      std::max(qnode->stat().numerator_used_error_, numerator_used_error_[q]);
    qnode->stat().numerator_n_pruned_ =
      std::min(qnode->stat().numerator_n_pruned_, numerator_n_pruned_[q]);
    

    // Refine summary statistics for the denominator.
  }
  
  // Clear postponed information for the numerator matrix.
  qnode->stat().postponed_numerator_l_.SetZero();
  qnode->stat().postponed_numerator_u_.SetZero();
  qnode->stat().postponed_numerator_used_error_ = 0;
  qnode->stat().postponed_numerator_n_pruned_ = 0;

  // Clear postponed information for the denominator matrix.
  qnode->stat().postponed_denominator_l_.SetZero();
  qnode->stat().postponed_denominator_u_.SetZero();
  qnode->stat().postponed_denominator_used_error_ = 0;
  qnode->stat().postponed_denominator_n_pruned_ = 0;  
}
