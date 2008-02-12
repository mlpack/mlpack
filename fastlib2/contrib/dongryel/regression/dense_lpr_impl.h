// Make sure this file is included only in dense_lpr.h. This is not a
// public header file!
#ifndef INSIDE_DENSE_LPR_H
#error "This file is not a public header file!"
#endif

#include "matrix_util.h"

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::SqdistAndKernelRanges_
(QueryTree *qnode, ReferenceTree *rnode,
 DRange &dsqd_range, DRange &kernel_value_range) {

  dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
  dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
  kernel_value_range = kernel_.RangeUnnormOnSq(dsqd_range);      
}

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::ResetQuery_(int q) {
  
  // First the numerator quantities.
  Vector q_numerator_l, q_numerator_e;
  numerator_l_.MakeColumnVector(q, &q_numerator_l);
  numerator_e_.MakeColumnVector(q, &q_numerator_e);
  q_numerator_l.SetZero();
  q_numerator_e.SetZero();
  numerator_used_error_[q] = 0;
  numerator_n_pruned_[q] = 0;
  
  // Then the denominator quantities,
  denominator_l_[q].SetZero();
  denominator_e_[q].SetZero();
  denominator_used_error_[q] = 0;
  denominator_n_pruned_[q] = 0;      
}

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::
ComputeTargetWeightedReferenceVectors_(ReferenceTree *rnode) {
  
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
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 2);
    rnode->stat().sum_target_weighted_data_alloc_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);
  }  
  else {
    
    // Recursively call the function with left and right and merge.
    ComputeTargetWeightedReferenceVectors_(rnode->left());
    ComputeTargetWeightedReferenceVectors_(rnode->right());
    
    la::AddOverwrite((rnode->left()->stat()).sum_target_weighted_data_,
		     (rnode->right()->stat()).sum_target_weighted_data_,
		     &(rnode->stat().sum_target_weighted_data_));
    rnode->stat().sum_target_weighted_data_error_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 2);
    rnode->stat().sum_target_weighted_data_alloc_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1); 
  }
}

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::
InitializeQueryTree_(QueryTree *qnode) {
    
  // Set the bounds to default values for the statistics.
  qnode->stat().SetZero();

  // If the query node is a leaf, then initialize the corresponding
  // bound quantities for each query point.
  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Reset the bounds corresponding to the particular query point.
      ResetQuery_(q);
    }
  }

  // Otherwise, then traverse to the left and the right.
  else {
    InitializeQueryTree_(qnode->left());
    InitializeQueryTree_(qnode->right());
  }
}

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::BestNodePartners_
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

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::BestNodePartners_
(ReferenceTree *nd, QueryTree *nd1, QueryTree *nd2, 
 QueryTree **partner1, QueryTree **partner2) {
  
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

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::DualtreeLprBase_
(QueryTree *qnode, ReferenceTree *rnode) {

  // Temporary variable for storing multivariate expansion of a
  // reference point.
  Vector reference_point_expansion;
  reference_point_expansion.Init(row_length_);

  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().numerator_norm_l_ = DBL_MAX;
  qnode->stat().numerator_used_error_ = 0;
  qnode->stat().numerator_n_pruned_ = DBL_MAX;
  qnode->stat().denominator_norm_l_ = DBL_MAX;
  qnode->stat().denominator_used_error_ = 0;
  qnode->stat().denominator_n_pruned_ = DBL_MAX;
  
  // Iterate over each query point.
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // Get the query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // Get the query-relevant quantities to be updated.
    double *q_numerator_l = numerator_l_.GetColumnPtr(q);
    double *q_numerator_e = numerator_e_.GetColumnPtr(q);

    // Incorporate the postponed information for the numerator vector.
    la::AddTo(row_length_, qnode->stat().postponed_numerator_l_.ptr(),
	      q_numerator_l);
    numerator_used_error_[q] += qnode->stat().postponed_numerator_used_error_;
    numerator_n_pruned_[q] += qnode->stat().postponed_numerator_n_pruned_;

    // Incorporate the postponed information for the denominator matrix.
    la::AddTo(qnode->stat().postponed_denominator_l_, &(denominator_l_[q]));
    denominator_used_error_[q] += 
      qnode->stat().postponed_denominator_used_error_;
    denominator_n_pruned_[q] += qnode->stat().postponed_denominator_n_pruned_;

    // Iterate over each reference point.
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the reference point and its training value.
      const double *r_col = rset_.GetColumnPtr(r);

      // Compute the reference point expansion.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order, r_col, reference_point_expansion.ptr());

      // Pairwise distance and kernel value and kernel value weighted
      // by the reference target training value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
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
				kernel_value * reference_point_expansion[i] *
				reference_point_expansion[j]);
	  denominator_e_[q].set(i, j, denominator_e_[q].get(i, j) +
				kernel_value * reference_point_expansion[i] *
				reference_point_expansion[j]);
	  
	} // End of iterating over each row.
      } // End of iterating over each column.

    } // End of iterating over each reference point.
    
    // Each query point has taken care of all reference points.
    numerator_n_pruned_[q] += 
      rnode->stat().sum_target_weighted_data_alloc_norm_;
    denominator_n_pruned_[q] +=
      rnode->stat().sum_data_outer_products_alloc_norm_;
    
    // Refine min summary statistics for the numerator.
    qnode->stat().numerator_norm_l_ =
      std::min(qnode->stat().numerator_norm_l_,
	       MatrixUtil::EntrywiseLpNorm(row_length_, q_numerator_l, 2));
    qnode->stat().numerator_used_error_ =
      std::max(qnode->stat().numerator_used_error_, numerator_used_error_[q]);
    qnode->stat().numerator_n_pruned_ =
      std::min(qnode->stat().numerator_n_pruned_, numerator_n_pruned_[q]);
    
    // Refine summary statistics for the denominator.
    qnode->stat().denominator_norm_l_ =
      std::min(qnode->stat().denominator_norm_l_,
	       MatrixUtil::EntrywiseLpNorm(denominator_l_[q], 2));
    qnode->stat().denominator_used_error_ =
      std::max(qnode->stat().denominator_used_error_, 
	       denominator_used_error_[q]);
    qnode->stat().denominator_n_pruned_ =
      std::min(qnode->stat().denominator_n_pruned_, denominator_n_pruned_[q]);

  } // End of iterating over each query point.
  
  // Clear postponed information for the numerator matrix.
  qnode->stat().postponed_numerator_l_.SetZero();
  qnode->stat().postponed_numerator_used_error_ = 0;
  qnode->stat().postponed_numerator_n_pruned_ = 0;

  // Clear postponed information for the denominator matrix.
  qnode->stat().postponed_denominator_l_.SetZero();
  qnode->stat().postponed_denominator_used_error_ = 0;
  qnode->stat().postponed_denominator_n_pruned_ = 0;  
}

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::DualtreeLprCanonical_
(QueryTree *qnode, ReferenceTree *rnode) {

  // Total amount of used error
  double numerator_used_error, denominator_used_error;
  
  // Total portion accounted by pruning.
  double numerator_n_pruned, denominator_n_pruned;

  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range, kernel_value_range;
  
  // Temporary variable for holding lower and estimate changes.
  Vector numerator_dl, numerator_de;
  numerator_dl.Init(row_length_);
  numerator_de.Init(row_length_);
  Matrix denominator_dl, denominator_de;
  denominator_dl.Init(row_length_, row_length_);
  denominator_de.Init(row_length_, row_length_);
  
  // Compute distance ranges and kernel ranges first.
  SqdistAndKernelRanges_(qnode, rnode, dsqd_range, kernel_value_range);
  
  // Try finite difference pruning first
  if(TPruneRule::Prunable
     (internal_relative_error_, 
      rroot_->stat().sum_target_weighted_data_alloc_norm_,
      rroot_->stat().sum_data_outer_products_alloc_norm_,
      qnode, rnode, dsqd_range, kernel_value_range,
      numerator_dl, numerator_de, numerator_used_error, numerator_n_pruned,
      denominator_dl, denominator_de, denominator_used_error,
      denominator_n_pruned)) {
    
    la::AddTo(numerator_dl, &(qnode->stat().postponed_numerator_l_));
    la::AddTo(numerator_de, &(qnode->stat().postponed_numerator_e_));
    qnode->stat().postponed_numerator_used_error_ += numerator_used_error;
    qnode->stat().postponed_numerator_n_pruned_ += numerator_n_pruned;
    
    la::AddTo(denominator_dl, &(qnode->stat().postponed_denominator_l_));
    la::AddTo(denominator_de, &(qnode->stat().postponed_denominator_e_));
    qnode->stat().postponed_denominator_used_error_ += denominator_used_error;
    qnode->stat().postponed_denominator_n_pruned_ += denominator_n_pruned;
    return;
  }

  // for leaf query node
  if(qnode->is_leaf()) {

    // for leaf pairs, go exhaustive
    if(rnode->is_leaf()) {
      DualtreeLprBase_(qnode, rnode);
      return;
    }

    // for non-leaf reference, expand reference node
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      BestNodePartners_(qnode, rnode->left(), rnode->right(), &rnode_first,
                        &rnode_second);
      DualtreeLprCanonical_(qnode, rnode_first);
      DualtreeLprCanonical_(qnode, rnode_second);
      return;
    }
  }
  
  // for non-leaf query node
  else {
    
    LprQStat &q_stat = qnode->stat();
    LprQStat &q_left_stat = qnode->left()->stat();
    LprQStat &q_right_stat = qnode->right()->stat();

    // Push down postponed bound changes owned by the current query
    // node to the children of the query node.
    la::AddTo(q_stat.postponed_numerator_l_, 
	      &q_left_stat.postponed_numerator_l_);
    la::AddTo(q_stat.postponed_numerator_l_,
	      &q_right_stat.postponed_numerator_l_);
    q_left_stat.postponed_numerator_used_error_ += 
      q_stat.postponed_numerator_used_error_;
    q_right_stat.postponed_numerator_used_error_ += 
      q_stat.postponed_numerator_used_error_;
    q_left_stat.postponed_numerator_n_pruned_ += 
      q_stat.postponed_numerator_n_pruned_;
    q_right_stat.postponed_numerator_n_pruned_ += 
      q_stat.postponed_numerator_n_pruned_;
    
    la::AddTo(q_stat.postponed_denominator_l_, 
	      &q_left_stat.postponed_denominator_l_);
    la::AddTo(q_stat.postponed_denominator_l_,
	      &q_right_stat.postponed_denominator_l_);
    q_left_stat.postponed_denominator_used_error_ += 
      q_stat.postponed_denominator_used_error_;
    q_right_stat.postponed_denominator_used_error_ += 
      q_stat.postponed_denominator_used_error_;
    q_left_stat.postponed_denominator_n_pruned_ += 
      q_stat.postponed_denominator_n_pruned_;
    q_right_stat.postponed_denominator_n_pruned_ += 
      q_stat.postponed_denominator_n_pruned_;

    // Clear the passed down postponed information.
    q_stat.postponed_numerator_l_.SetZero();
    q_stat.postponed_numerator_used_error_ = 0;
    q_stat.postponed_numerator_n_pruned_ = 0;
    q_stat.postponed_denominator_l_.SetZero();
    q_stat.postponed_denominator_used_error_ = 0;
    q_stat.postponed_denominator_n_pruned_ = 0;
    
    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      QueryTree *qnode_first = NULL, *qnode_second = NULL;
      
      BestNodePartners_(rnode, qnode->left(), qnode->right(), &qnode_first,
			&qnode_second);
      DualtreeLprCanonical_(qnode_first, rnode);
      DualtreeLprCanonical_(qnode_second, rnode);
    }
    
    // for non-leaf reference node, expand both query and reference nodes
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      
      BestNodePartners_(qnode->left(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeLprCanonical_(qnode->left(), rnode_first);
      DualtreeLprCanonical_(qnode->left(), rnode_second);
      
      BestNodePartners_(qnode->right(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeLprCanonical_(qnode->right(), rnode_first);
      DualtreeLprCanonical_(qnode->right(), rnode_second);
    }
    
    // reaccumulate the summary statistics.
    q_stat.numerator_norm_l_ = 
      std::min
      (q_left_stat.numerator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_left_stat.postponed_numerator_l_, 2),
       q_right_stat.numerator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_right_stat.postponed_numerator_l_, 2));
    q_stat.numerator_used_error_ = 
      std::max(q_left_stat.numerator_used_error_,
	       q_right_stat.numerator_used_error_);
    q_stat.numerator_n_pruned_ = 
      std::min(q_left_stat.numerator_n_pruned_,
	       q_right_stat.numerator_n_pruned_);
    q_stat.denominator_norm_l_ = 
      std::min
      (q_left_stat.denominator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_left_stat.postponed_denominator_l_, 2),
       q_right_stat.denominator_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_right_stat.postponed_denominator_l_, 2));
    q_stat.denominator_used_error_ = 
      std::max(q_left_stat.denominator_used_error_,
	       q_right_stat.denominator_used_error_);
    q_stat.denominator_n_pruned_ = 
      std::min(q_left_stat.denominator_n_pruned_,
	       q_right_stat.denominator_n_pruned_);    

    return;
  } // end of the case: non-leaf query node.  
}

template<typename TKernel, int lpr_order, typename TPruneRule>
void DenseLpr<TKernel, lpr_order, TPruneRule>::
FinalizeQueryTree_(QueryTree *qnode) {
  
  LprQStat &q_stat = qnode->stat();

  if(qnode->is_leaf()) {

    Matrix pseudoinverse_denominator;
    pseudoinverse_denominator.Init(row_length_, row_length_);
    Vector least_squares_solution;
    least_squares_solution.Init(row_length_);
    Vector query_point_expansion;
    query_point_expansion.Init(row_length_);

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Get the query point.
      const double *query_point = qset_.GetColumnPtr(q);

      // Get the numerator vectors accumulating the sums to update.
      Vector q_numerator_l, q_numerator_e;
      numerator_l_.MakeColumnVector(q, &q_numerator_l);
      numerator_e_.MakeColumnVector(q, &q_numerator_e);

      // Incorporate the postponed information for the numerator.
      la::AddTo(q_stat.postponed_numerator_l_, &q_numerator_l);
      la::AddTo(q_stat.postponed_numerator_e_, &q_numerator_e);

      // Incorporate the postponed information for the denominator.
      la::AddTo(q_stat.postponed_denominator_l_, &(denominator_l_[q]));
      la::AddTo(q_stat.postponed_denominator_e_, &(denominator_e_[q]));

      // After incorporating all of the postponed information,
      // finalize the regression estimate by solving the appropriate
      // linear system (B^T W(q) B) z(q) = B^T W(q) Y for z(q) and
      // taking the dot product between z(q) and the polynomial power
      // formed from the query point coordinates.
      MatrixUtil::PseudoInverse(denominator_e_[q], &pseudoinverse_denominator);
      la::MulOverwrite(pseudoinverse_denominator, q_numerator_e,
		       &least_squares_solution);
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order, query_point, query_point_expansion.ptr());
      regression_estimates_[q] = la::Dot(query_point_expansion,
					 least_squares_solution);
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
