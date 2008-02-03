// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeRightHandSidesBase_
(Tree *qnode, Tree *rnode) {
  
  // for each query point
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // get query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // get the column vectors accumulating the sums to update.
    const double *q_right_hand_sides_l_ = right_hand_sides_l_.GetColumnPtr(q);
    const double *q_right_hand_sides_e_ = right_hand_sides_e_.GetColumnPtr(q);
    const double *q_right_hand_sides_u_ = right_hand_sides_u_.GetColumnPtr(q);

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
	
	q_right_hand_sides_l_[d] += kernel_value * r_weights[d];
	q_right_hand_sides_e_[d] += kernel_value * r_weights[d];
	q_right_hand_sides_u_[d] += kernel_value * r_weights[d];

      } // end of iterating over each vector component.
      
    } // end of iterating over each reference point.

    // Now, loop over each vector component for the current query and
    // correct the upper bound by the assumption made in the
    // initialization phase of the query tree. Refine min and max
    // summary statistics.
    for(index_t d = 0; d <= dimension_; d++) {
      
      // Correct the upper bound for the current query first.
      q_right_hand_sides_u_[d] -= (rnode->sum_targets_weighted_by_data_)[d];
      
      // Refine bounds.
      qnode->stat().right_hand_sides_l_[d] =
	std::min(qnode->stat().right_hand_sides_l_[d], 
		 q_right_hand_sides_l_[d]);
      qnode->stat().right_hand_sides_u_[d] =
	std::max(qnode->stat().right_hand_sides_u_[d],
		 q_right_hand_sides_u_[d]);

    } // end of looping over each vector component.

  } // end of iterating over each query point.

  // Clear postponed information.
  qnode->stat().postponed_right_hand_sides_l_.SetZero();
  qnode->stat().postponed_right_hand_sides_u_.SetZero();
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeRightHandSidesCanonical_
(Tree *qnode, Tree *rnode) {
  
}
