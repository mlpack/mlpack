// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
bool LocalLinearKrylov<TKernel>::PrunableRightHandSides_
(Tree *qnode, Tree *rnode, DRange &dsqd_range, DRange &kernel_value_range) {
  
  return true;
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeRightHandSidesBase_
(Tree *qnode, Tree *rnode) {
  
  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  (qnode->stat().right_hand_sides_l_).SetAll(DBL_MAX);
  (qnode->stat().right_hand_sides_u_).SetAll(-DBL_MAX);
  
  // for each query point
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // get query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // get the column vectors accumulating the sums to update.
    double *q_right_hand_sides_l = right_hand_sides_l_.GetColumnPtr(q);
    double *q_right_hand_sides_e = right_hand_sides_e_.GetColumnPtr(q);
    double *q_right_hand_sides_u = right_hand_sides_u_.GetColumnPtr(q);

    // Incorporate the postponed information.
    la::AddTo(row_length_, 
	      (qnode->stat().postponed_right_hand_sides_l_).ptr(),
	      q_right_hand_sides_l);
    la::AddTo(row_length_, 
	      (qnode->stat().postponed_right_hand_sides_u_).ptr(),
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
      (qnode->stat().right_hand_sides_l_)[d] =
	std::min((qnode->stat().right_hand_sides_l_)[d], 
		 q_right_hand_sides_l[d]);
      (qnode->stat().right_hand_sides_u_)[d] =
	std::max((qnode->stat().right_hand_sides_u_)[d],
		 q_right_hand_sides_u[d]);

    } // end of looping over each vector component.

  } // end of iterating over each query point.

  // Clear postponed information.
  (qnode->stat().postponed_right_hand_sides_l_).SetZero();
  (qnode->stat().postponed_right_hand_sides_u_).SetZero();
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeRightHandSidesCanonical_
(Tree *qnode, Tree *rnode) {
      
  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;
  
  // try finite difference pruning first
  if(PrunableRightHandSides_(qnode, rnode, dsqd_range, kernel_value_range)) {
    la::AddTo(right_hand_sides_l_change_,
	      &(qnode->stat().postponed_right_hand_sides_l_));
    la::AddTo(right_hand_sides_e_change_,
	      &(qnode->stat().postponed_right_hand_sides_e_));
    la::AddTo(right_hand_sides_u_change_,
	      &(qnode->stat().postponed_right_hand_sides_u_));
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
    la::AddTo(qnode->stat().postponed_right_hand_sides_l_,
	      &((qnode->left()->stat()).postponed_right_hand_sides_l_));
    la::AddTo(qnode->stat().postponed_right_hand_sides_l_,
	      &((qnode->right()->stat()).postponed_right_hand_sides_l_));
    la::AddTo(qnode->stat().postponed_right_hand_sides_u_,
	      &((qnode->left()->stat()).postponed_right_hand_sides_u_));
    la::AddTo(qnode->stat().postponed_right_hand_sides_u_,
	      &((qnode->right()->stat()).postponed_right_hand_sides_u_));
    (qnode->stat().postponed_right_hand_sides_l_).SetZero();
    (qnode->stat().postponed_right_hand_sides_u_).SetZero();

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
      (qnode->stat().right_hand_sides_l_)[d] =
	std::min(((qnode->left()->stat()).right_hand_sides_l_)[d] +
		 ((qnode->left()->stat()).postponed_right_hand_sides_l_)[d],
		 ((qnode->right()->stat()).right_hand_sides_l_)[d] +
		 ((qnode->right()->stat()).postponed_right_hand_sides_l_)[d]);
      (qnode->stat().right_hand_sides_u_)[d] =
	std::max(((qnode->left()->stat()).right_hand_sides_u_)[d] +
		 ((qnode->left()->stat()).postponed_right_hand_sides_u_)[d],
		 ((qnode->right()->stat()).right_hand_sides_u_)[d] +
		 ((qnode->right()->stat()).postponed_right_hand_sides_u_)[d]);
    }
    return;
  } // end of the case: non-leaf query node.
}
