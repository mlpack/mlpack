// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void LocalLinearKrylov<TKernel>::InitializeQueryTreeRightHandSides_
(Tree *qnode) {
  
  // Set the bounds to default values.
  (qnode->stat().ll_vector_l_).SetZero();
  (qnode->stat().ll_vector_u_).CopyValues
    (rroot_->stat().sum_targets_weighted_by_data_);
  (qnode->stat().postponed_ll_vector_l_).SetZero();
  (qnode->stat().postponed_ll_vector_e_).SetZero();
  (qnode->stat().postponed_ll_vector_u_).SetZero();

  // If the query node is a leaf, then initialize the corresponding
  // bound quantities for each query point.
  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      Vector q_right_hand_sides_l, q_right_hand_sides_e, q_right_hand_sides_u;

      vector_l_.MakeColumnVector(q, &q_right_hand_sides_l);
      vector_e_.MakeColumnVector(q, &q_right_hand_sides_e);
      vector_u_.MakeColumnVector(q, &q_right_hand_sides_u);
      
      q_right_hand_sides_l.SetZero();
      q_right_hand_sides_e.SetZero();
      q_right_hand_sides_u.CopyValues
	(rroot_->stat().sum_targets_weighted_by_data_);
    }
  }

  // Otherwise, then traverse to the left and the right.
  else {
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

      // Tally up the weighted targets.
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

template<typename TKernel>
bool LocalLinearKrylov<TKernel>::PrunableRightHandSides_
(Tree *qnode, Tree *rnode, DRange &dsqd_range, DRange &kernel_value_range,
 double &used_error) {
      
  // try pruning after bound refinement: first compute distance/kernel
  // value bounds
  dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
  dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
  kernel_value_range = kernel_.RangeUnnormOnSq(dsqd_range);

  // Compute the vector component lower and upper bound changes. This
  // assumes that the maximum kernel value is 1.
  la::ScaleOverwrite(kernel_value_range.lo,
		     rnode->stat().sum_targets_weighted_by_data_,
		     &vector_l_change_);
  la::ScaleOverwrite(0.5 * (kernel_value_range.lo +
			    kernel_value_range.hi),
		     rnode->stat().sum_targets_weighted_by_data_,
		     &vector_e_change_);
  la::ScaleOverwrite((kernel_value_range.hi - 1.0),
		     rnode->stat().sum_targets_weighted_by_data_,
		     &vector_u_change_);

  // Refine the lower bound based on the current postponed lower bound
  // change and the newly gained refinement due to comparing the
  // current query and reference node pair.
  la::AddOverwrite(qnode->stat().ll_vector_l_,
		   qnode->stat().postponed_ll_vector_l_,
		   &new_vector_l_);
  la::AddTo(vector_l_change_, &new_vector_l_);

  // Compute the L1 norm of the most refined lower bound.
  double l1_norm_new_right_hand_sides_l_ = L1Norm_(new_vector_l_);
    
  // Compute the allowed amount of error for pruning the given query
  // and reference pair.
  double allowed_err = 
    (relative_error_ * (rnode->stat().l1_norm_sum_targets_weighted_by_data_) *
     l1_norm_new_right_hand_sides_l_) / 
    (rroot_->stat().l1_norm_sum_targets_weighted_by_data_);

  used_error = 0.5 * kernel_value_range.width() * 
    (rnode->stat().l1_norm_sum_targets_weighted_by_data_);
  
  // check pruning condition  
  return (used_error <= allowed_err);
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeRightHandSidesBase_
(Tree *qnode, Tree *rnode) {
  
  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  (qnode->stat().ll_vector_l_).SetAll(DBL_MAX);
  (qnode->stat().ll_vector_u_).SetAll(-DBL_MAX);
  
  // for each query point
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // get query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // get the column vectors accumulating the sums to update.
    double *q_right_hand_sides_l = vector_l_.GetColumnPtr(q);
    double *q_right_hand_sides_e = vector_e_.GetColumnPtr(q);
    double *q_right_hand_sides_u = vector_u_.GetColumnPtr(q);

    // Incorporate the postponed information.
    la::AddTo(row_length_, 
	      (qnode->stat().postponed_ll_vector_l_).ptr(),
	      q_right_hand_sides_l);
    la::AddTo(row_length_, 
	      (qnode->stat().postponed_ll_vector_u_).ptr(),
	      q_right_hand_sides_u);

    // for each reference point
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {

      // get reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // get the column vector containing the appropriate weights.
      const double *r_weights = 
	rset_targets_weighted_by_coordinates_.GetColumnPtr(r);

      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernel_.EvalUnnormOnSq(dsqd);

      // for each vector component, update the lower/estimate/upper
      // bound quantities.
      for(index_t d = 0; d <= dimension_; d++) {
	
	q_right_hand_sides_l[d] += kernel_value * r_weights[d];
	q_right_hand_sides_e[d] += kernel_value * r_weights[d];
	q_right_hand_sides_u[d] += kernel_value * r_weights[d];

      } // end of iterating over each vector component.
      
    } // end of iterating over each reference point.

    // Now, loop over each vector component for the current query and
    // correct the upper bound by the assumption made in the
    // initialization phase of the query tree. Refine min and max
    // summary statistics.
    for(index_t d = 0; d <= dimension_; d++) {
      
      // Correct the upper bound for the current query first.
      q_right_hand_sides_u[d] -= 
	(rnode->stat().sum_targets_weighted_by_data_)[d];
      
      // Refine bounds.
      (qnode->stat().ll_vector_l_)[d] =
	std::min((qnode->stat().ll_vector_l_)[d], 
		 q_right_hand_sides_l[d]);
      (qnode->stat().ll_vector_u_)[d] =
	std::max((qnode->stat().ll_vector_u_)[d],
		 q_right_hand_sides_u[d]);

    } // end of looping over each vector component.

  } // end of iterating over each query point.

  // Clear postponed information.
  (qnode->stat().postponed_ll_vector_l_).SetZero();
  (qnode->stat().postponed_ll_vector_u_).SetZero();
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeRightHandSidesCanonical_
(Tree *qnode, Tree *rnode) {
  
  // Total amount of used error
  double used_error;

  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;
  
  // try finite difference pruning first
  if(PrunableRightHandSides_(qnode, rnode, dsqd_range, kernel_value_range,
			     used_error)) {
    la::AddTo(vector_l_change_,
	      &(qnode->stat().postponed_ll_vector_l_));
    la::AddTo(vector_e_change_,
	      &(qnode->stat().postponed_ll_vector_e_));
    la::AddTo(vector_u_change_,
	      &(qnode->stat().postponed_ll_vector_u_));
    num_finite_difference_prunes_++;
    return;
  }
  
  // for leaf query node
  if(qnode->is_leaf()) {
    
    // for leaf pairs, go exhaustive
    if(rnode->is_leaf()) {
      DualtreeRightHandSidesBase_(qnode, rnode);
      return;
    }
    
    // for non-leaf reference, expand reference node
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      BestNodePartners_(qnode, rnode->left(), rnode->right(), &rnode_first,
			&rnode_second);
      DualtreeRightHandSidesCanonical_(qnode, rnode_first);
      DualtreeRightHandSidesCanonical_(qnode, rnode_second);
      return;
    }
  }
  
  // for non-leaf query node
  else {
    
    // Push down postponed bound changes owned by the current query
    // node to the children of the query node and clear them.
    la::AddTo(qnode->stat().postponed_ll_vector_l_,
	      &((qnode->left()->stat()).postponed_ll_vector_l_));
    la::AddTo(qnode->stat().postponed_ll_vector_l_,
	      &((qnode->right()->stat()).postponed_ll_vector_l_));
    la::AddTo(qnode->stat().postponed_ll_vector_u_,
	      &((qnode->left()->stat()).postponed_ll_vector_u_));
    la::AddTo(qnode->stat().postponed_ll_vector_u_,
	      &((qnode->right()->stat()).postponed_ll_vector_u_));
    (qnode->stat().postponed_ll_vector_l_).SetZero();
    (qnode->stat().postponed_ll_vector_u_).SetZero();

    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      Tree *qnode_first = NULL, *qnode_second = NULL;
      
      BestNodePartners_(rnode, qnode->left(), qnode->right(), &qnode_first,
			&qnode_second);
      DualtreeRightHandSidesCanonical_(qnode_first, rnode);
      DualtreeRightHandSidesCanonical_(qnode_second, rnode);
    }
    
    // for non-leaf reference node, expand both query and reference nodes
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      
      BestNodePartners_(qnode->left(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeRightHandSidesCanonical_(qnode->left(), rnode_first);
      DualtreeRightHandSidesCanonical_(qnode->left(), rnode_second);
      
      BestNodePartners_(qnode->right(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeRightHandSidesCanonical_(qnode->right(), rnode_first);
      DualtreeRightHandSidesCanonical_(qnode->right(), rnode_second);
    }
    
    // reaccumulate the summary statistics.
    for(index_t d = 0; d <= dimension_; d++) {
      (qnode->stat().ll_vector_l_)[d] =
	std::min(((qnode->left()->stat()).ll_vector_l_)[d] +
		 ((qnode->left()->stat()).postponed_ll_vector_l_)[d],
		 ((qnode->right()->stat()).ll_vector_l_)[d] +
		 ((qnode->right()->stat()).postponed_ll_vector_l_)[d]);
      (qnode->stat().ll_vector_u_)[d] =
	std::max(((qnode->left()->stat()).ll_vector_u_)[d] +
		 ((qnode->left()->stat()).postponed_ll_vector_u_)[d],
		 ((qnode->right()->stat()).ll_vector_u_)[d] +
		 ((qnode->right()->stat()).postponed_ll_vector_u_)[d]);
    }
    return;
  } // end of the case: non-leaf query node.
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::FinalizeQueryTreeRightHandSides_
(Tree *qnode) {

  LocalLinearKrylovStat &q_stat = qnode->stat();

  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      
      // Get the column vectors accumulating the sums to update.
      double *q_right_hand_sides_l = vector_l_.GetColumnPtr(q);
      double *q_right_hand_sides_e = vector_e_.GetColumnPtr(q);
      double *q_right_hand_sides_u = vector_u_.GetColumnPtr(q);
      
      // Incorporate the postponed information.
      la::AddTo(row_length_,
		(q_stat.postponed_ll_vector_l_).ptr(),
		q_right_hand_sides_l);
      la::AddTo(row_length_,
		(q_stat.postponed_ll_vector_e_).ptr(),
		q_right_hand_sides_e);
      la::AddTo(row_length_,
		(q_stat.postponed_ll_vector_u_).ptr(),
		q_right_hand_sides_u);

      // Normalize.
      la::Scale(row_length_, 1.0 / ((double) rset_.n_cols()), 
		q_right_hand_sides_l);
      la::Scale(row_length_, 1.0 / ((double) rset_.n_cols()),
		q_right_hand_sides_e);
      la::Scale(row_length_, 1.0 / ((double) rset_.n_cols()),
		q_right_hand_sides_u);
    }
  }
  else {
    
    LocalLinearKrylovStat &q_left_stat = qnode->left()->stat();
    LocalLinearKrylovStat &q_right_stat = qnode->right()->stat();

    // Push down approximations
    la::AddTo(q_stat.postponed_ll_vector_l_,
	      &(q_left_stat.postponed_ll_vector_l_));
    la::AddTo(q_stat.postponed_ll_vector_l_,
	      &(q_right_stat.postponed_ll_vector_l_));
    la::AddTo(q_stat.postponed_ll_vector_e_,
              &(q_left_stat.postponed_ll_vector_e_));
    la::AddTo(q_stat.postponed_ll_vector_e_,
              &(q_right_stat.postponed_ll_vector_e_));
    la::AddTo(q_stat.postponed_ll_vector_u_,
              &(q_left_stat.postponed_ll_vector_u_));
    la::AddTo(q_stat.postponed_ll_vector_u_,
              &(q_right_stat.postponed_ll_vector_u_));

    FinalizeQueryTreeRightHandSides_(qnode->left());
    FinalizeQueryTreeRightHandSides_(qnode->right());
  }
}
