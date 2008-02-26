// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

#include "multi_index_util.h"
#include "matrix_util.h"

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::InitializeQueryTree_(QueryTree *qnode) {
  
  // Set the bounds to default values.
  qnode->stat().Reset();

  // If the query node is not a leaf, then recurse.
  if(!qnode->is_leaf()) {

    InitializeQueryTree_(qnode->left());
    InitializeQueryTree_(qnode->right());
  }
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::InitializeReferenceStatistics_
(ReferenceTree *rnode, int column_index, const Vector &weights) {
  
  if(rnode->is_leaf()) {
    
    // Temporary vector for computing the reference point expansion.
    Vector reference_point_expansion;
    reference_point_expansion.Init(row_length_);

    // Clear the sum statistics before accumulating.
    rnode->stat().Reset();

    // For a leaf reference node, iterate over each reference point
    // and compute the weighted vector and tally these up for the
    // sum statistics owned by the reference node.
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // Get the pointer to the current reference point.
      Vector r_col;
      rset_.MakeColumnVector(r, &r_col);

      // Get the pointer to the reference column to be updated.
      double *r_target_weighted_by_coordinates = 
	target_weighted_rset_.GetColumnPtr(r);

      // Compute the multiindex expansion of the given reference point.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, r_col.ptr(), reference_point_expansion.ptr());

      // Scale the expansion by the reference target.
      la::ScaleOverwrite
	(row_length_, weights[r] * reference_point_expansion[column_index], 
	 reference_point_expansion.ptr(), r_target_weighted_by_coordinates);
      
      // Accumulate the far field coefficient for the target weighted
      // reference vector and the outerproduct.
      for(index_t j = 0; j < row_length_; j++) {
	rnode->stat().target_weighted_data_far_field_expansion_[j].
	  Add(r_target_weighted_by_coordinates[j], kernels_[r].bandwidth_sq(),
	      r_col);
      }

      // Tally up the weighted targets.
      la::AddTo(row_length_, r_target_weighted_by_coordinates,
		(rnode->stat().sum_target_weighted_data_).ptr());
      
      // Accumulate the bandwidth statistics.
      rnode->stat().min_bandwidth_kernel.Init
	(std::min(sqrt(rnode->stat().min_bandwidth_kernel.bandwidth_sq()),
		  sqrt(kernels_[r].bandwidth_sq())));
      rnode->stat().max_bandwidth_kernel.Init
	(std::max(sqrt(rnode->stat().max_bandwidth_kernel.bandwidth_sq()),
		  sqrt(kernels_[r].bandwidth_sq())));
    } // end of iterating over each reference point.

    // Compute L1 norm of the accumulated sum
    rnode->stat().sum_target_weighted_data_error_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);
    rnode->stat().sum_target_weighted_data_alloc_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);
  }  
  else {
    
    // Recursively call the function with left and right and merge.
    InitializeReferenceStatistics_(rnode->left(), column_index, weights);
    InitializeReferenceStatistics_(rnode->right(), column_index, weights);
   
    // Compute the sum of the sub sums.
    la::AddOverwrite((rnode->left()->stat()).sum_target_weighted_data_,
		     (rnode->right()->stat()).sum_target_weighted_data_,
		     &(rnode->stat().sum_target_weighted_data_));
    rnode->stat().sum_target_weighted_data_error_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);
    rnode->stat().sum_target_weighted_data_alloc_norm_ =
      MatrixUtil::EntrywiseLpNorm(rnode->stat().sum_target_weighted_data_, 1);

    // Translate far-field moments of the child to form the parent.
    for(index_t j = 0; j < row_length_; j++) {
      rnode->stat().target_weighted_data_far_field_expansion_[j].
	Add(rnode->left()->stat().
	    target_weighted_data_far_field_expansion_[j]);
      rnode->stat().target_weighted_data_far_field_expansion_[j].
	Add(rnode->right()->stat().
	    target_weighted_data_far_field_expansion_[j]);
    } // end of iterating over each column.

    // Compute the min of the min bandwidths and the max of the max
    // bandwidths owned among the children.
    rnode->stat().min_bandwidth_kernel.Init
      (std::min
       (sqrt(rnode->left()->stat().min_bandwidth_kernel.bandwidth_sq()),
	sqrt(rnode->right()->stat().min_bandwidth_kernel.bandwidth_sq())));
    rnode->stat().max_bandwidth_kernel.Init
      (std::max
       (sqrt(rnode->left()->stat().max_bandwidth_kernel.bandwidth_sq()),
	sqrt(rnode->right()->stat().max_bandwidth_kernel.bandwidth_sq())));
  }
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::DualtreeWeightedVectorSumBase_
(QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
 Matrix &right_hand_sides_l, Matrix &right_hand_sides_e,
 Vector &right_hand_sides_used_error, Vector &right_hand_sides_n_pruned) {
  
  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().ll_vector_norm_l_ = DBL_MAX;
  qnode->stat().ll_vector_used_error_ = 0;
  qnode->stat().ll_vector_n_pruned_ = DBL_MAX;
  
  // for each query point
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // get query point.
    const double *q_col = qset.GetColumnPtr(q);

    // get the column vectors accumulating the sums to update.
    double *q_right_hand_side_l = right_hand_sides_l.GetColumnPtr(q);
    double *q_right_hand_side_e = right_hand_sides_e.GetColumnPtr(q);

    // Incorporate the postponed information.
    la::AddTo(row_length_, (qnode->stat().postponed_ll_vector_l_).ptr(),
	      q_right_hand_side_l);
    right_hand_sides_used_error[q] += 
      qnode->stat().postponed_ll_vector_used_error_;
    right_hand_sides_n_pruned[q] += 
      qnode->stat().postponed_ll_vector_n_pruned_;

    // for each reference point
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {

      // get reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // get the column vector containing the appropriate weights.
      const double *r_weights = target_weighted_rset_.GetColumnPtr(r);

      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[r].EvalUnnormOnSq(dsqd);

      // For each vector component, update the lower/estimate bound
      // quantities.
      la::AddExpert(row_length_, kernel_value, r_weights, q_right_hand_side_l);
      la::AddExpert(row_length_, kernel_value, r_weights, q_right_hand_side_e);
      
    } // end of iterating over each reference point.

    // The current query point now has taken care of all reference
    // points.
    right_hand_sides_n_pruned[q] += 
      rnode->stat().sum_target_weighted_data_alloc_norm_;
    
    // Refine the lower bound on the L1-norm of the vector sum, the
    // used error and the portion of the reference set pruned for the
    // query points.
    qnode->stat().ll_vector_norm_l_ =
      std::min
      (qnode->stat().ll_vector_norm_l_,
       MatrixUtil::EntrywiseLpNorm(row_length_, q_right_hand_side_l, 1));
    qnode->stat().ll_vector_used_error_ =
      std::max(qnode->stat().ll_vector_used_error_, 
	       right_hand_sides_used_error[q]);
    qnode->stat().ll_vector_n_pruned_ =
      std::min(qnode->stat().ll_vector_n_pruned_, 
	       right_hand_sides_n_pruned[q]);

  } // end of iterating over each query point.

  // Clear postponed information.
  qnode->stat().postponed_ll_vector_l_.SetZero();
  qnode->stat().postponed_ll_vector_used_error_ = 0;
  qnode->stat().postponed_ll_vector_n_pruned_ = 0;
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::DualtreeWeightedVectorSumCanonical_
(QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
 Matrix &right_hand_sides_l, Matrix &right_hand_sides_e, 
 Vector &right_hand_sides_used_error, Vector &right_hand_sides_n_pruned) {
  
  // Variables for storing changes due to a prune.
  double delta_used_error, delta_n_pruned;
  Vector delta_l, delta_e;
  delta_l.Init(row_length_);
  delta_e.Init(row_length_);

  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;
  
  // Compute the squared distance and kernel value ranges.
  LprUtil::SqdistAndKernelRanges_
    (qnode, rnode, dsqd_range, kernel_value_range);

  // try finite difference pruning first
  if(TPruneRule::PrunableWeightedVectorSum
     (internal_relative_error_, 
      rnode->stat().sum_target_weighted_data_alloc_norm_,
      qnode, rnode, dsqd_range, kernel_value_range, delta_l, delta_e,
      delta_used_error, delta_n_pruned)) {
    la::AddTo(delta_l, &(qnode->stat().postponed_ll_vector_l_));
    la::AddTo(delta_e, &(qnode->stat().postponed_ll_vector_e_));
    qnode->stat().postponed_ll_vector_used_error_ += delta_used_error;
    qnode->stat().postponed_ll_vector_n_pruned_ += delta_n_pruned;
    num_finite_difference_prunes_++;
    return;
  }
  
  // For the Epanechnikov kernel, we can prune using the far field
  // moments if the maximum distance between the two nodes is within
  // the bandwidth! This if-statement does not apply to the Gaussian
  // kernel, so I need to fix in the future!
  if(rnode->stat().min_bandwidth_kernel.bandwidth_sq() >= dsqd_range.hi && 
     rnode->count() > dimension_ * dimension_) {

    for(index_t j = 0; j < row_length_; j++) {
      
      qnode->stat().postponed_ll_vector_l_[j] += 
	rnode->stat().target_weighted_data_far_field_expansion_[j].
	ComputeMinKernelSum(qnode->bound());
      qnode->stat().postponed_moment_ll_vector_e_[j].
	Add(rnode->stat().target_weighted_data_far_field_expansion_[j]);
    }

    qnode->stat().postponed_ll_vector_n_pruned_ += delta_n_pruned;

    // Keep track of the far-field prunes.
    num_epanechnikov_prunes_++;

    return;
  }

  // for leaf query node
  if(qnode->is_leaf()) {
    
    // for leaf pairs, go exhaustive
    if(rnode->is_leaf()) {
      DualtreeWeightedVectorSumBase_
	(qnode, rnode, qset, right_hand_sides_l, right_hand_sides_e, 
	 right_hand_sides_used_error, right_hand_sides_n_pruned);
      return;
    }
    
    // for non-leaf reference, expand reference node
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      LprUtil::BestReferenceNodePartners(qnode, rnode->left(), rnode->right(), 
					 &rnode_first, &rnode_second);
      DualtreeWeightedVectorSumCanonical_
	(qnode, rnode_first, qset, right_hand_sides_l, right_hand_sides_e, 
	 right_hand_sides_used_error, right_hand_sides_n_pruned);
      DualtreeWeightedVectorSumCanonical_
	(qnode, rnode_second, qset, right_hand_sides_l, right_hand_sides_e, 
	 right_hand_sides_used_error, right_hand_sides_n_pruned);
      return;
    }
  }
  
  // for non-leaf query node
  else {
    
    // Declare references to the query stats.
    KrylovLprQStat<TKernel> &q_stat = qnode->stat();
    KrylovLprQStat<TKernel> &q_left_stat = qnode->left()->stat();
    KrylovLprQStat<TKernel> &q_right_stat = qnode->right()->stat();

    // Push down postponed bound changes owned by the current query
    // node to the children of the query node
    la::AddTo(q_stat.postponed_ll_vector_l_,
	      &(q_left_stat.postponed_ll_vector_l_));
    la::AddTo(q_stat.postponed_ll_vector_l_,
	      &(q_right_stat.postponed_ll_vector_l_));
    q_left_stat.postponed_ll_vector_used_error_ +=
      q_stat.postponed_ll_vector_used_error_;
    q_right_stat.postponed_ll_vector_used_error_ +=
      q_stat.postponed_ll_vector_used_error_;
    q_left_stat.postponed_ll_vector_n_pruned_ +=
      q_stat.postponed_ll_vector_n_pruned_;
    q_right_stat.postponed_ll_vector_n_pruned_ +=
      q_stat.postponed_ll_vector_n_pruned_;

    // Clear the statistics after pushing them downwards.
    q_stat.postponed_ll_vector_l_.SetZero();
    q_stat.postponed_ll_vector_used_error_ = 0;
    q_stat.postponed_ll_vector_n_pruned_ = 0;

    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      QueryTree *qnode_first = NULL, *qnode_second = NULL;
      
      LprUtil::BestQueryNodePartners(rnode, qnode->left(), qnode->right(), 
				     &qnode_first, &qnode_second);
      DualtreeWeightedVectorSumCanonical_
	(qnode_first, rnode, qset, right_hand_sides_l, right_hand_sides_e, 
	 right_hand_sides_used_error, right_hand_sides_n_pruned);
      DualtreeWeightedVectorSumCanonical_
	(qnode_second, rnode, qset, right_hand_sides_l, right_hand_sides_e, 
	 right_hand_sides_used_error, right_hand_sides_n_pruned);
    }
    
    // for non-leaf reference node, expand both query and reference nodes
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      
      LprUtil::BestReferenceNodePartners(qnode->left(), rnode->left(), 
					 rnode->right(),
					 &rnode_first, &rnode_second);
      DualtreeWeightedVectorSumCanonical_
	(qnode->left(), rnode_first, qset, right_hand_sides_l, 
	 right_hand_sides_e, right_hand_sides_used_error, 
	 right_hand_sides_n_pruned);
      DualtreeWeightedVectorSumCanonical_
	(qnode->left(), rnode_second, qset, right_hand_sides_l, 
	 right_hand_sides_e, right_hand_sides_used_error, 
	 right_hand_sides_n_pruned);
      
      LprUtil::BestReferenceNodePartners(qnode->right(), rnode->left(), 
					 rnode->right(),
					 &rnode_first, &rnode_second);
      DualtreeWeightedVectorSumCanonical_
	(qnode->right(), rnode_first, qset, right_hand_sides_l, 
	 right_hand_sides_e, right_hand_sides_used_error, 
	 right_hand_sides_n_pruned);
      DualtreeWeightedVectorSumCanonical_
	(qnode->right(), rnode_second, qset, right_hand_sides_l, 
	 right_hand_sides_e, right_hand_sides_used_error, 
	 right_hand_sides_n_pruned);
    }

    // reaccumulate the summary statistics.
    q_stat.ll_vector_norm_l_ =
      std::min(q_left_stat.ll_vector_norm_l_ +
	       MatrixUtil::EntrywiseLpNorm
	       (q_left_stat.postponed_ll_vector_l_, 1),
	       q_right_stat.ll_vector_norm_l_ +
	       MatrixUtil::EntrywiseLpNorm
	       (q_right_stat.postponed_ll_vector_l_, 1));
    q_stat.ll_vector_used_error_ =
      std::max(q_left_stat.ll_vector_used_error_ +
	       q_left_stat.postponed_ll_vector_used_error_,
	       q_right_stat.ll_vector_used_error_ +
	       q_right_stat.postponed_ll_vector_used_error_);
    q_stat.ll_vector_n_pruned_ =
      std::min(q_left_stat.ll_vector_n_pruned_ +
	       q_left_stat.postponed_ll_vector_n_pruned_,
	       q_right_stat.ll_vector_n_pruned_ +
	       q_right_stat.postponed_ll_vector_n_pruned_);
    return;
  } // end of the case: non-leaf query node.
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::FinalizeQueryTree_
(QueryTree *qnode, const Matrix &qset, 
 Matrix &right_hand_sides_l, Matrix &right_hand_sides_e,
 Vector &right_hand_sides_used_error, Vector &right_hand_sides_n_pruned) {

  KrylovLprQStat<TKernel> &q_stat = qnode->stat();

  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Get the current query point.
      Vector q_col;
      qset.MakeColumnVector(q, &q_col);

      // Get the column vectors accumulating the sums to update.
      double *q_right_hand_sides_l = right_hand_sides_l.GetColumnPtr(q);
      double *q_right_hand_sides_e = right_hand_sides_e.GetColumnPtr(q);
      
      // Incorporate the postponed information.
      la::AddTo(row_length_, (q_stat.postponed_ll_vector_l_).ptr(),
		q_right_hand_sides_l);
      la::AddTo(row_length_, (q_stat.postponed_ll_vector_e_).ptr(),
		q_right_hand_sides_e);

      // Incorporate the postponed estimates using the Epanechnikov
      // series expansion.
      for(index_t i = 0; i < row_length_; i++) {
	q_right_hand_sides_e[i] += 
	  qnode->stat().postponed_moment_ll_vector_e_[i].
	  ComputeKernelSum(q_col);
      }

      right_hand_sides_used_error[q] += q_stat.postponed_ll_vector_used_error_;
      right_hand_sides_n_pruned[q] += q_stat.postponed_ll_vector_n_pruned_;
    }
  }
  else {
    
    KrylovLprQStat<TKernel> &q_left_stat = qnode->left()->stat();
    KrylovLprQStat<TKernel> &q_right_stat = qnode->right()->stat();

    // Push down approximations
    la::AddTo(q_stat.postponed_ll_vector_l_,
	      &(q_left_stat.postponed_ll_vector_l_));
    la::AddTo(q_stat.postponed_ll_vector_l_,
	      &(q_right_stat.postponed_ll_vector_l_));
    la::AddTo(q_stat.postponed_ll_vector_e_,
              &(q_left_stat.postponed_ll_vector_e_));
    la::AddTo(q_stat.postponed_ll_vector_e_,
              &(q_right_stat.postponed_ll_vector_e_));
    q_left_stat.postponed_ll_vector_used_error_ +=
      q_stat.postponed_ll_vector_used_error_;
    q_right_stat.postponed_ll_vector_used_error_ += 
      q_stat.postponed_ll_vector_used_error_;
    q_left_stat.postponed_ll_vector_n_pruned_ +=
      q_stat.postponed_ll_vector_n_pruned_;
    q_right_stat.postponed_ll_vector_n_pruned_ +=
      q_stat.postponed_ll_vector_n_pruned_;

    // Push down Epanechnikov series expansion pruning.
    for(index_t i = 0; i < row_length_; i++) {
      q_left_stat.postponed_moment_ll_vector_e_[i].Add
	(q_stat.postponed_moment_ll_vector_e_[i]);
      q_right_stat.postponed_moment_ll_vector_e_[i].Add
	(q_stat.postponed_moment_ll_vector_e_[i]);
    }

    FinalizeQueryTree_(qnode->left(), qset, right_hand_sides_l, 
		       right_hand_sides_e, right_hand_sides_used_error, 
		       right_hand_sides_n_pruned);
    FinalizeQueryTree_(qnode->right(), qset, right_hand_sides_l, 
		       right_hand_sides_e, right_hand_sides_used_error, 
		       right_hand_sides_n_pruned);
  }
}
