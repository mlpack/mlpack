// Make sure this file is included only in dense_lpr.h. This is not a
// public header file!
#ifndef INSIDE_DENSE_LPR_H
#error "This file is not a public header file!"
#endif

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::Reset_(int q) {
  
  // First the numerator quantities.
  Vector q_numerator_l, q_numerator_e;
  numerator_l_.MakeColumnVector(q, &q_numerator_l);
  numerator_e_.MakeColumnVector(q, &q_numerator_e);
  q_numerator_l.SetZero();
  q_numerator_e.SetZero();
  numerator_used_error_ = 0;
  numerator_n_pruned_ = 0;
  
  // Then the denominator quantities,
  denominator_l_[q].SetZero();
  denominator_e_[q].SetZero();
  denominator_used_error_ = 0;
  denominator_n_pruned_ = 0;      
}

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
    rnode->stat().sum_target_weighted_data_error_norm_ =
      MatrixUtil::FrobeniusNorm(rnode->stat().sum_target_weighted_data_);
  }  
  else {
    
    // Recursively call the function with left and right and merge.
    ComputeTargetWeightedReferenceVectors_(rnode->left());
    ComputeTargetWeightedReferenceVectors_(rnode->right());
    
    la::AddOverwrite((rnode->left()->stat()).sum_target_weighted_data_,
		     (rnode->right()->stat()).sum_target_weighted_data_,
		     &(rnode->stat().sum_target_weighted_data_));
    rnode->stat().sum_target_weighted_data_error_norm_ =
      MatrixUtil::FrobeniusNorm(rnode->stat().sum_target_weighted_data_);
  }
}

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::InitializeQueryTree_(QueryTree *qnode) {
    
  // Set the bounds to default values for the statistics.
  qnode->stat().SetZero();

  // If the query node is a leaf, then initialize the corresponding
  // bound quantities for each query point.
  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Reset the bounds corresponding to the particular query point.
      Reset_(q);
    }
  }

  // Otherwise, then traverse to the left and the right.
  else {
    InitializeQueryTree_(qnode->left());
    InitializeQueryTree_(qnode->right());
  }
}

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::BestNodePartners_
(QueryTree *nd, ReferenceTree *nd1, ReferenceTree *nd2, 
 ReferenceTree **partner1, ReferenceTree **partner2) {
  
  double d1 = nd->bound().MinDistanceSq(nd1->bound());
  double d2 = nd->bound().MinDistanceSq(nd2->bound());
  
  if(d1 <= d2) {
    *partner1 = nd1;
    *partner2 = nd2;
  }
  else {
    *partner1 = nd2;
    *partner2 = nd1;
  }
}

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::DualtreeLprBase_
(QueryTree *qnode, ReferenceTree *rnode) {

  // Temporary variable for storing multivariate expansion of a
  // reference point.
  Vector reference_point_expansion;
  reference_point_expansion.Init(row_length_);

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().numerator_l_.SetAll(DBL_MAX);
  qnode->stat().numerator_used_error_ = 0;
  qnode->stat().numerator_n_pruned_ = DBL_MAX;
  qnode->stat().denominator_l_.SetAll(DBL_MAX);
  qnode->stat().denominator_used_error_ = 0;
  qnode->stat().denominator_n_pruned_ = DBL_MAX;
  
  // compute unnormalized sum
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // Get the query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // Get the query-relevant quantities to be updated.
    double *q_numerator_l = numerator_l_.GetColumnPtr(q);
    double *q_numerator_e = numerator_e_.GetColumnPtr(q);

    // Incorporate the postponed information for the numerator vector.
    la::AddTo(row_length_, qnode->stat().postponed_numerator_l_,
	      q_numerator_l);
    numerator_used_error_[q] += qnode->stat().postponed_numerator_used_error_;
    numerator_n_pruned_[q] += qnode->stat().postponed_n_pruned_;

    // Incorporate the postponed information for the denominator matrix.
    la::AddTo(qnode->stat().postponed_denominator_l_, &(denominator_l_[q]));
    denominator_used_error_[q] += 
      qnode->stat().postponed_denominator_used_error_;
    denominator_n_pruned_[q] += qnode->stat().postponed_denominator_n_pruned_;

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
	
	// Loop over each row of the matrix to be updated.
	for(index_t i = 0; i < row_length_; i++) {
	  
	  // Tally the sum up for the denominator matrix B^T W(q) B.
	  denominator_l_[q].set(i, j, denominator_l_[q].get(i, j) +
				kernel_value *
				reference_point_expansion[i] *
				reference_point_expansion[j]);
	  denominator_e_[q].set(i, j, denominator_e_[q].get(i, j) +
				kernel_value * 
				reference_point_expansion[i] *
				reference_point_expansion[j]);
	  
	} // End of iterating over each row.
      } // End of iterating over each column.

    } // End of iterating over each reference point.
    
    // Each query point has taken care of all reference points.
    numerator_n_pruned_[q] += 
      rnode->stat().sum_target_weighted_data_alloc_norm_;
    denominator_n_pruned_[q] +=
      rnode->stat().sum_data_outer_products_alloc_norm_;
    
    // Refine min and max summary statistics for the numerator.

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

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::DualtreeLprCanonical_
(QueryTree *qnode, ReferenceTree *rnode) {

  // Total amount of used error
  double numerator_used_error, denominator_used_error;
  
  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;
  
  // try finite difference pruning first
  /*
  if(DualtreeLpr::Prunable_(qnode, rnode, dsqd_range, kernel_value_range,
			    used_error, denominator_used_error)) {
    la::AddTo(numerator_l_change_,
              &(qnode->stat().postponed_numerator_l_));
    la::AddTo(numerator_e_change_,
              &(qnode->stat().postponed_numerator_e_));
    la::AddTo(numerator_u_change_,
              &(qnode->stat().postponed_numerator_u_));
    num_finite_difference_prunes_++;
    return;
  }
  */

  // for leaf query node
  if(qnode->is_leaf()) {

    // for leaf pairs, go exhaustive
    if(rnode->is_leaf()) {
      DualtreeRightHandSidesBase_(qnode, rnode);
      return;
    }

    // for non-leaf reference, expand reference node
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      BestNodePartners_(qnode, rnode->left(), rnode->right(), &rnode_first,
                        &rnode_second);
      DualtreeRightHandSidesCanonical_(qnode, rnode_first);
      DualtreeRightHandSidesCanonical_(qnode, rnode_second);
      return;
    }
  }
}

template<typename TKernel, int lpr_order>
void DenseLpr<TKernel, lpr_order>::FinalizeQueryTree_(QueryTree *qnode) {
  
  LprQStat &q_stat = qnode->stat();

  if(qnode->ls_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Get the numerator vectors accumulating the sums to update.
      double *q_numerator_l = numerator_l_.GetColumnPtr(q);
      double *q_numerator_e = numerator_e_.GetColumnPtr(q);

      // Incorporate the postponed information for the numerator.
      la::AddTo(row_length_, q_stat.postponed_numerator_l_.ptr(),
		q_numerator_l);
      la::AddTo(row_length_, q_stat.postponed_numerator_e_.ptr(),
		q_numerator_e);

      // Get the denominator matrices accumulating the sums to update.
      Matrix &q_denominator_l = denominator_l_[q];
      Matrix &q_denominator_e = denominator_e_[q];

      // Incorporate the postponed information for the denominator.
      la::AddTo(q_stat.postponed_denominator_l_, &q_denominator_l);
      la::AddTo(q_stat.postponed_denominator_e_, &q_denominator_e);
    }
  }
  else {
    
    LprQStat &q_left_stat = qnode->left()->stat();
    LprQStat &q_right_stat = qnode->right()->stat();

    // Push down approximations for the numerator.
    la::AddTo(q_stat.postponed_numerator_l_,
	      &(q_left_stat.postponed_numerator_l_));
    la::AddTo(q_stat.postponed_numerator_e_,
	      &(q_left_stat.postponed_numerator_e_));
    la::AddTo(q_stat.postponed_numerator_l_,
              &(q_right_stat.postponed_numerator_l_));
    la::AddTo(q_stat.postponed_numerator_e_,
              &(q_right_stat.postponed_numerator_e_));
    
    // Push down approximations for the denominator.
    la::AddTo(q_stat.postponed_denominator_l_,
              &(q_left_stat.postponed_denominator_l_));
    la::AddTo(q_stat.postponed_denominator_e_,
              &(q_left_stat.postponed_denominator_e_));
    la::AddTo(q_stat.postponed_denominator_l_,
              &(q_right_stat.postponed_denominator_l_));
    la::AddTo(q_stat.postponed_denominator_e_,
              &(q_right_stat.postponed_denominator_e_));
    
    FinalizeQueryTree_(qnode->left());
    FinalizeQueryTree_(qnode->right());
  }
}
