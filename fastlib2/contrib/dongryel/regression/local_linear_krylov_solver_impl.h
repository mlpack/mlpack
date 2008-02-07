// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeSolverBase_
(Tree *qnode, Tree *rnode, Matrix &current_lanczos_vectors,
 DRange &root_negative_dot_product_range,
 DRange &root_positive_dot_product_range) {
  
  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  (qnode->stat().ll_vector_l_).SetAll(DBL_MAX);
  (qnode->stat().ll_vector_u_).SetAll(-DBL_MAX);
  (qnode->stat().neg_ll_vector_l_).SetAll(DBL_MAX);
  (qnode->stat().neg_ll_vector_u_).SetAll(-DBL_MAX);
  
  // for each query point
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    
    // get query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // Get the query point's associated current Lanczos vector.
    const double *q_lanczos_vector = current_lanczos_vectors.GetColumnPtr(q);

    // get the column vectors accumulating the sums to update.
    double *q_vector_l = vector_l_.GetColumnPtr(q);
    double *q_vector_e = vector_e_.GetColumnPtr(q);
    double *q_vector_u = vector_u_.GetColumnPtr(q);
    double *q_neg_vector_l = neg_vector_l_.GetColumnPtr(q);
    double *q_neg_vector_e = neg_vector_e_.GetColumnPtr(q);
    double *q_neg_vector_u = neg_vector_u_.GetColumnPtr(q);

    // Incorporate the postponed information.
    la::AddTo(row_length_, (qnode->stat().postponed_ll_vector_l_).ptr(),
	      q_vector_l);
    la::AddTo(row_length_, (qnode->stat().postponed_ll_vector_u_).ptr(),
	      q_vector_u);
    la::AddTo(row_length_, (qnode->stat().postponed_neg_ll_vector_l_).ptr(),
	      q_neg_vector_l);
    la::AddTo(row_length_, (qnode->stat().postponed_neg_ll_vector_u_).ptr(),
	      q_neg_vector_u);
    
    // for each reference point
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // get reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernel_.EvalUnnormOnSq(dsqd);

      // Take the dot product between the query point's Lanczos vector
      // and [1 r^T]^T.
      double dot_product = q_lanczos_vector[0];
      for(index_t d = 1; d <= dimension_; d++) {
	dot_product += r_col[d - 1] * q_lanczos_vector[d];
      }
      double front_factor = dot_product * kernel_value;

      // For each vector component, update the lower/estimate/upper
      // bound quantities.
      if(front_factor > 0) {
	q_vector_l[0] += front_factor;
	q_vector_e[0] += front_factor;
	q_vector_u[0] += front_factor;
      }
      else {
	q_neg_vector_l[0] += front_factor;
	q_neg_vector_e[0] += front_factor;
	q_neg_vector_u[0] += front_factor;
      }      
      for(index_t d = 1; d <= dimension_; d++) {
	
	if(front_factor > 0) {
	  q_vector_l[d] += front_factor * r_col[d - 1];
	  q_vector_e[d] += front_factor * r_col[d - 1];
	  q_vector_u[d] += front_factor * r_col[d - 1];
	}
	else {
	  q_neg_vector_l[d] += front_factor * r_col[d - 1];
	  q_neg_vector_e[d] += front_factor * r_col[d - 1];
	  q_neg_vector_u[d] += front_factor * r_col[d - 1];
	}
      } // end of iterating over each vector component.
      
    } // end of iterating over each reference point.

    // Now, loop over each vector component for the current query and
    // correct the upper bound by the assumption made in the
    // initialization phase of the query tree. Refine min and max
    // summary statistics.
    for(index_t d = 0; d <= dimension_; d++) {
      
      // Correct the upper bound for the current query first.
      q_vector_u[d] -= (root_positive_dot_product_range.hi *
			(rnode->stat().sum_coordinates_)[d]);
      q_neg_vector_l[d] -= (root_negative_dot_product_range.lo *
			    (rnode->stat().sum_coordinates_)[d]);
      
      // Refine bounds.
      (qnode->stat().ll_vector_l_)[d] =
	std::min((qnode->stat().ll_vector_l_)[d], 
		 q_vector_l[d]);
      (qnode->stat().ll_vector_u_)[d] =
	std::max((qnode->stat().ll_vector_u_)[d],
		 q_vector_u[d]);
      (qnode->stat().neg_ll_vector_l_)[d] =
	std::min((qnode->stat().neg_ll_vector_l_)[d], 
		 q_neg_vector_l[d]);
      (qnode->stat().neg_ll_vector_u_)[d] =
	std::max((qnode->stat().neg_ll_vector_u_)[d],
		 q_neg_vector_u[d]);
      
    } // end of looping over each vector component.

  } // end of iterating over each query point.

  // Clear postponed information.
  (qnode->stat().postponed_ll_vector_l_).SetZero();
  (qnode->stat().postponed_ll_vector_u_).SetZero();
  (qnode->stat().postponed_neg_ll_vector_l_).SetZero();
  (qnode->stat().postponed_neg_ll_vector_u_).SetZero();
}

template<typename TKernel>
bool LocalLinearKrylov<TKernel>::PrunableSolver_
(Tree *qnode, Tree *rnode, Matrix &current_lanczos_vectors,
 DRange &root_negative_dot_product_range, 
 DRange &root_positive_dot_product_range, DRange &dsqd_range, 
 DRange &kernel_value_range, double &used_error) {
  
  // Temporary variables hold the dot product ranges.
  DRange negative_dot_product_range, positive_dot_product_range;
  
  // Compute the dot product range.
  DotProductBetweenTwoBounds_(qnode, rnode, negative_dot_product_range,
			      positive_dot_product_range);

  // try pruning after bound refinement: first compute distance/kernel
  // value bounds
  dsqd_range.lo = qnode->bound().MinDistanceSq(rnode->bound());
  dsqd_range.hi = qnode->bound().MaxDistanceSq(rnode->bound());
  kernel_value_range = kernel_.RangeUnnormOnSq(dsqd_range);

  // Compute the vector component lower and upper bound changes. This
  // assumes that the maximum kernel value is 1.
  la::ScaleOverwrite(positive_dot_product_range.lo * kernel_value_range.lo,
		     rnode->stat().sum_coordinates_,
		     &vector_l_change_);
  la::ScaleOverwrite(0.5 * (positive_dot_product_range.lo *
			    kernel_value_range.lo +
			    positive_dot_product_range.hi *
			    kernel_value_range.hi),
		     rnode->stat().sum_coordinates_,
		     &vector_e_change_);
  la::ScaleOverwrite(positive_dot_product_range.hi * kernel_value_range.hi - 
		     root_positive_dot_product_range.hi,
		     rnode->stat().sum_coordinates_,
		     &vector_u_change_);

  la::ScaleOverwrite(negative_dot_product_range.lo * kernel_value_range.hi - 
		     root_negative_dot_product_range.lo,
		     rnode->stat().sum_coordinates_,
		     &neg_vector_u_change_);
  la::ScaleOverwrite(0.5 * (negative_dot_product_range.lo *
			    kernel_value_range.hi +
			    negative_dot_product_range.hi *
			    kernel_value_range.lo),
		     rnode->stat().sum_coordinates_,
		     &neg_vector_e_change_);
  la::ScaleOverwrite(negative_dot_product_range.hi * kernel_value_range.lo,
		     rnode->stat().sum_coordinates_,
		     &neg_vector_l_change_);

  // Refine the positive lower bound based on the current postponed
  // lower bound change and the newly gained refinement due to
  // comparing the current query and reference node pair. Do the same
  // for the negative upper bound.
  la::AddOverwrite(qnode->stat().ll_vector_l_,
		   qnode->stat().postponed_ll_vector_l_,
		   &new_vector_l_);
  la::AddTo(vector_l_change_, &new_vector_l_);
  la::AddOverwrite(qnode->stat().neg_ll_vector_u_,
		   qnode->stat().postponed_neg_ll_vector_u_,
		   &new_neg_vector_u_);
  la::AddTo(neg_vector_u_change_, &new_neg_vector_u_);

  // Compute the L1 norm of the most refined lower bound.
  double l1_norm_vector_l = L1Norm_(new_vector_l_);
  double l1_norm_neg_vector_u = L1Norm_(new_neg_vector_u_);
  
  // Compute the allowed amount of error for pruning the given query
  // and reference pair.
  double allowed_err = 
    (relative_error_ * (rnode->stat().l1_norm_sum_coordinates_) *
     (l1_norm_vector_l + l1_norm_neg_vector_u)) / 
    (rroot_->stat().l1_norm_sum_coordinates_);

  used_error = 0.5 * ((positive_dot_product_range.hi *
		       kernel_value_range.hi -
		       positive_dot_product_range.lo *
		       kernel_value_range.lo) +
		      (negative_dot_product_range.hi *
		       kernel_value_range.lo -
		       negative_dot_product_range.lo *
		       kernel_value_range.hi)) *
    rnode->stat().l1_norm_sum_coordinates_;
  
  // check pruning condition  
  return (used_error <= allowed_err);
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DualtreeSolverCanonical_
(Tree *qnode, Tree *rnode, Matrix &current_lanczos_vectors,
 DRange &root_negative_dot_product_range,
 DRange &root_positive_dot_product_range) {
    
  // Total amount of used error
  double used_error;
  
  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;
  
  // try finite difference pruning first
  if(PrunableSolver_(qnode, rnode, current_lanczos_vectors,
		     root_negative_dot_product_range,
		     root_positive_dot_product_range,
		     dsqd_range, kernel_value_range, used_error)) {

    la::AddTo(vector_l_change_,
	      &(qnode->stat().postponed_ll_vector_l_));
    la::AddTo(vector_e_change_,
	      &(qnode->stat().postponed_ll_vector_e_));
    la::AddTo(vector_u_change_,
	      &(qnode->stat().postponed_ll_vector_u_));
    la::AddTo(neg_vector_l_change_,
	      &(qnode->stat().postponed_neg_ll_vector_l_));
    la::AddTo(neg_vector_e_change_,
	      &(qnode->stat().postponed_neg_ll_vector_e_));
    la::AddTo(neg_vector_u_change_,
	      &(qnode->stat().postponed_neg_ll_vector_u_));

    num_finite_difference_prunes_++;
    return;
  }
  
  // for leaf query node
  if(qnode->is_leaf()) {
    
    // for leaf pairs, go exhaustive
    if(rnode->is_leaf()) {
      DualtreeSolverBase_(qnode, rnode, current_lanczos_vectors,
			  root_negative_dot_product_range,
			  root_positive_dot_product_range);
      return;
    }
    
    // for non-leaf reference, expand reference node
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      BestNodePartners_(qnode, rnode->left(), rnode->right(), &rnode_first,
			&rnode_second);
      DualtreeSolverCanonical_(qnode, rnode_first, current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
      DualtreeSolverCanonical_(qnode, rnode_second, current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
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
      DualtreeSolverCanonical_(qnode_first, rnode, current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
      DualtreeSolverCanonical_(qnode_second, rnode, current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
    }
    
    // for non-leaf reference node, expand both query and reference nodes
    else {
      Tree *rnode_first = NULL, *rnode_second = NULL;
      
      BestNodePartners_(qnode->left(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeSolverCanonical_(qnode->left(), rnode_first,
			       current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
      DualtreeSolverCanonical_(qnode->left(), rnode_second,
			       current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
      
      BestNodePartners_(qnode->right(), rnode->left(), rnode->right(),
			&rnode_first, &rnode_second);
      DualtreeSolverCanonical_(qnode->right(), rnode_first,
			       current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
      DualtreeSolverCanonical_(qnode->right(), rnode_second,
			       current_lanczos_vectors,
			       root_negative_dot_product_range,
			       root_positive_dot_product_range);
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
      (qnode->stat().neg_ll_vector_l_)[d] =
	std::min(((qnode->left()->stat()).neg_ll_vector_l_)[d] +
		 ((qnode->left()->stat()).postponed_neg_ll_vector_l_)[d],
		 ((qnode->right()->stat()).neg_ll_vector_l_)[d] +
		 ((qnode->right()->stat()).postponed_neg_ll_vector_l_)[d]);
      (qnode->stat().neg_ll_vector_u_)[d] =
	std::max(((qnode->left()->stat()).neg_ll_vector_u_)[d] +
		 ((qnode->left()->stat()).postponed_neg_ll_vector_u_)[d],
		 ((qnode->right()->stat()).neg_ll_vector_u_)[d] +
		 ((qnode->right()->stat()).postponed_neg_ll_vector_u_)[d]);
    }
    return;
  } // end of the case: non-leaf query node.
  
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DotProductBetweenTwoBounds_
(Tree *qnode, Tree *rnode, DRange &negative_dot_product_range,
 DRange &positive_dot_product_range) {
  
  DHrectBound<2> lanczos_vectors_bound = qnode->stat().lanczos_vectors_bound_;

  // Initialize the dot-product ranges.
  negative_dot_product_range.lo = negative_dot_product_range.hi = 0;
  positive_dot_product_range.lo = positive_dot_product_range.hi = 0;
  
  if(lanczos_vectors_bound.get(0).lo > 0) {
    positive_dot_product_range.hi += lanczos_vectors_bound.get(0).hi;
  }
  else if(lanczos_vectors_bound.get(0).Contains(0)) {
    positive_dot_product_range.hi += lanczos_vectors_bound.get(0).hi;
    negative_dot_product_range.lo += lanczos_vectors_bound.get(0).lo;
  }
  else {
    negative_dot_product_range.lo += lanczos_vectors_bound.get(0).lo;
    negative_dot_product_range.hi += lanczos_vectors_bound.get(0).hi;
  }

  for(index_t d = 1; d <= dimension_; d++) {

    const DRange &lanczos_directional_bound = lanczos_vectors_bound.get(d);
    const DRange &reference_node_directional_bound = rnode->bound().get(d - 1);
    
    if(lanczos_directional_bound.lo > 0) {
      positive_dot_product_range.hi += lanczos_directional_bound.hi *
	reference_node_directional_bound.hi;
    }
    else if(lanczos_directional_bound.Contains(0)) {
      positive_dot_product_range.hi += lanczos_directional_bound.hi *
	reference_node_directional_bound.hi;
      negative_dot_product_range.lo += lanczos_directional_bound.lo *
	reference_node_directional_bound.hi;
    }
    else {
      negative_dot_product_range.lo += lanczos_directional_bound.lo *
	reference_node_directional_bound.hi;
      negative_dot_product_range.hi += lanczos_directional_bound.hi *
	reference_node_directional_bound.lo;
    }
  } // End of looping over each component...
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::InitializeQueryTreeSolutionBound_
(Tree *qnode, Matrix &current_lanczos_vectors) {

  // If the query node is a leaf, then exhaustively iterate over and
  // form bounding boxes of the current solution.
  if(qnode->is_leaf()) {
    (qnode->stat().lanczos_vectors_bound_).Reset();
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      Vector lanczos_vector;
      current_lanczos_vectors.MakeColumnVector(q, &lanczos_vector);
      qnode->stat().lanczos_vectors_bound_ |= lanczos_vector;
    }
  }

  // Otherwise, traverse the left and the right and combine the
  // bounding boxes of the solutions for the two children.
  else {    
    InitializeQueryTreeSolutionBound_(qnode->left(), current_lanczos_vectors);
    InitializeQueryTreeSolutionBound_(qnode->right(), current_lanczos_vectors);
    
    (qnode->stat().lanczos_vectors_bound_).Reset();
    qnode->stat().lanczos_vectors_bound_ |= 
      (qnode->left()->stat()).lanczos_vectors_bound_;
    qnode->stat().lanczos_vectors_bound_ |=
      (qnode->right()->stat()).lanczos_vectors_bound_;
  }
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::InitializeQueryTreeSumBound_
(Tree *qnode, DRange &root_negative_dot_product_range,
 DRange &root_positive_dot_product_range) {
  
  // Set the bounds to default values.
  (qnode->stat().ll_vector_l_).SetZero();
  la::ScaleOverwrite(root_positive_dot_product_range.hi,
		     rroot_->stat().sum_coordinates_,
		     &(qnode->stat().ll_vector_u_));
  
  la::ScaleOverwrite(root_negative_dot_product_range.lo,
		     rroot_->stat().sum_coordinates_,
		     &(qnode->stat().neg_ll_vector_l_));
  (qnode->stat().neg_ll_vector_u_).SetZero();
  
  // Set the postponed quantities to zero.
  (qnode->stat().postponed_ll_vector_l_).SetZero();
  (qnode->stat().postponed_ll_vector_e_).SetZero();
  (qnode->stat().postponed_ll_vector_u_).SetZero();
  (qnode->stat().postponed_neg_ll_vector_l_).SetZero();
  (qnode->stat().postponed_neg_ll_vector_e_).SetZero();
  (qnode->stat().postponed_neg_ll_vector_u_).SetZero();

  // If the query node is a leaf, then initialize the corresponding
  // bound statistics for each query point.
  if(qnode->is_leaf()) {

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      Vector q_vector_l, q_vector_e, q_vector_u;
      Vector q_neg_vector_l, q_neg_vector_e, q_neg_vector_u;
      
      vector_l_.MakeColumnVector(q, &q_vector_l);
      vector_e_.MakeColumnVector(q, &q_vector_e);
      vector_u_.MakeColumnVector(q, &q_vector_u);
      neg_vector_l_.MakeColumnVector(q, &q_neg_vector_l);
      neg_vector_e_.MakeColumnVector(q, &q_neg_vector_e);
      neg_vector_u_.MakeColumnVector(q, &q_neg_vector_u);
      
      q_vector_l.SetZero();
      q_vector_e.SetZero();
      q_vector_u.CopyValues(qnode->stat().ll_vector_u_);
      
      q_neg_vector_l.CopyValues(qnode->stat().neg_ll_vector_l_);
      q_neg_vector_e.SetZero();
      q_neg_vector_u.SetZero();
    }
  }
  else {
    
    InitializeQueryTreeSumBound_(qnode->left(), 
				 root_negative_dot_product_range,
				 root_positive_dot_product_range);
    InitializeQueryTreeSumBound_(qnode->right(), 
				 root_negative_dot_product_range,
				 root_positive_dot_product_range);
  }
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::FinalizeQueryTreeLanczosMultiplier_
(Tree *qnode) {

  LocalLinearKrylovStat &q_stat = qnode->stat();

  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      
      // Get the column vectors accumulating the sums to update.
      double *q_vector_l = vector_l_.GetColumnPtr(q);
      double *q_vector_e = vector_e_.GetColumnPtr(q);
      double *q_vector_u = vector_u_.GetColumnPtr(q);
      double *q_neg_vector_l = neg_vector_l_.GetColumnPtr(q);
      double *q_neg_vector_e = neg_vector_e_.GetColumnPtr(q);
      double *q_neg_vector_u = neg_vector_u_.GetColumnPtr(q);
      
      // Incorporate the postponed information.
      la::AddTo(row_length_, (q_stat.postponed_ll_vector_l_).ptr(),
		q_vector_l);
      la::AddTo(row_length_, (q_stat.postponed_ll_vector_e_).ptr(),
		q_vector_e);
      la::AddTo(row_length_, (q_stat.postponed_ll_vector_u_).ptr(),
		q_vector_u);
      la::AddTo(row_length_, (q_stat.postponed_neg_ll_vector_l_).ptr(),
		q_neg_vector_l);
      la::AddTo(row_length_, (q_stat.postponed_neg_ll_vector_e_).ptr(),
		q_neg_vector_e);
      la::AddTo(row_length_, (q_stat.postponed_neg_ll_vector_u_).ptr(),
		q_neg_vector_u);
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
    
    la::AddTo(q_stat.postponed_neg_ll_vector_l_,
	      &(q_left_stat.postponed_neg_ll_vector_l_));
    la::AddTo(q_stat.postponed_neg_ll_vector_l_,
	      &(q_right_stat.postponed_neg_ll_vector_l_));
    la::AddTo(q_stat.postponed_neg_ll_vector_e_,
              &(q_left_stat.postponed_neg_ll_vector_e_));
    la::AddTo(q_stat.postponed_neg_ll_vector_e_,
              &(q_right_stat.postponed_neg_ll_vector_e_));
    la::AddTo(q_stat.postponed_neg_ll_vector_u_,
              &(q_left_stat.postponed_neg_ll_vector_u_));
    la::AddTo(q_stat.postponed_neg_ll_vector_u_,
              &(q_right_stat.postponed_neg_ll_vector_u_));

    FinalizeQueryTreeLanczosMultiplier_(qnode->left());
    FinalizeQueryTreeLanczosMultiplier_(qnode->right());
  }
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::SolveLeastSquaresByKrylov_() {

  // Initialize the initial solutions to zero vectors.
  solution_vectors_e_.SetZero();

  // Temporary variables needed for SYMMLQ iteration...
  Matrix previous_lanczos_vectors;
  Matrix current_lanczos_vectors;
  Matrix v_tilde_mat;
  previous_lanczos_vectors.Init(row_length_, qset_.n_cols()); 
  v_tilde_mat.Init(row_length_, qset_.n_cols());

  Vector g_double_tilde_vec;
  Vector g_vec;
  g_double_tilde_vec.Init(qset_.n_cols());
  g_vec.Init(qset_.n_cols());
  
  current_lanczos_vectors.Init(row_length_, qset_.n_cols());

  // More temporary variables for SYMMLQ routine...
  Vector c_vec, beta_vec, beta_tilde_vec, s_vec;
  Matrix w_mat;
  c_vec.Init(qset_.n_cols());
  beta_vec.Init(qset_.n_cols());
  beta_tilde_vec.Init(qset_.n_cols());
  s_vec.Init(qset_.n_cols());
  w_mat.Init(row_length_, qset_.n_cols());
  
  // Initialize before entering the main iteration... This
  // initialization implicitly assumes that initial guess to the
  // linear system is the zero vector.
  current_lanczos_vectors.CopyValues(vector_e_);
  NormalizeMatrixColumnVectors_(current_lanczos_vectors, g_double_tilde_vec);
  beta_vec.SetZero();
  beta_tilde_vec.SetZero();
  c_vec.SetAll(-1);
  s_vec.SetZero();
  previous_lanczos_vectors.SetZero();
  w_mat.CopyValues(current_lanczos_vectors);
  g_vec.SetZero();

  // Temporary variables to hold dot product ranges for the root nodes
  // of the two trees.
  DRange root_negative_dot_product_range, root_positive_dot_product_range;
    
  // Main iteration of the SYMMLQ algorithm - repeat until
  // "convergence"...
  for(index_t m = 0; m < sqrt(row_length_); m++) {

    // Initialize the query tree solution bounds.
    InitializeQueryTreeSolutionBound_(qroot_, current_lanczos_vectors);
    
    // Compute the dot product bounds.
    DotProductBetweenTwoBounds_(qroot_, rroot_, 
				root_negative_dot_product_range,
				root_positive_dot_product_range);
    
    // Initialize the query tree bound statistics.
    InitializeQueryTreeSumBound_(qroot_, root_negative_dot_product_range,
				 root_positive_dot_product_range);

    // Multiply the current lanczos vector with the linear operator.
    DualtreeSolverCanonical_(qroot_, rroot_, current_lanczos_vectors,
			     root_negative_dot_product_range,
			     root_positive_dot_product_range);
    FinalizeQueryTreeLanczosMultiplier_(qroot_);

    // Compute v_tilde_mat (the residue after applying the linear
    // operator the current Lanczos vector).
    la::AddOverwrite(vector_e_, neg_vector_e_, &v_tilde_mat);
    /*
    printf("Finished multiplying..\n");

    TestKrylovComputation_(v_tilde_mat, current_lanczos_vectors);
    exit(0);
    */

    for(index_t q = 0; q < qset_.n_cols(); q++) {
      double *v_tilde_mat_column = v_tilde_mat.GetColumnPtr(q);
      double *previous_lanczos_vector = 
	previous_lanczos_vectors.GetColumnPtr(q);
      double *current_lanczos_vector =
	current_lanczos_vectors.GetColumnPtr(q);

      la::AddExpert(row_length_, -beta_vec[q], previous_lanczos_vector,
		    v_tilde_mat_column);


      // Compute alpha (a dot product b etween the current Lanczos
      // vector and v_tilde vector).
      double alpha = la::Dot(row_length_, current_lanczos_vector,
			     v_tilde_mat_column);
      
      // Subtract the component of the current Lanczos vector (a form
      // of Gram-Schmidt orthogonalization.)
      la::AddExpert(row_length_, -alpha, current_lanczos_vector,
		    v_tilde_mat_column);
      
      // Compute the length of v_tilde_mat_column and store into
      // beta_vec.
      beta_vec[q] = la::LengthEuclidean(row_length_, v_tilde_mat_column);

      // Make a backup copy of the current Lanczos vector.
      previous_lanczos_vectors.CopyValues(current_lanczos_vectors);
      
      // Set a new current Lanczos vector based on v_tilde_mat_column
      if(beta_vec[q] > 0) {
	la::ScaleOverwrite(row_length_, 1.0 / beta_vec[q], v_tilde_mat_column,
			   current_lanczos_vector);
      }
      else {
	la::ScaleOverwrite(row_length_, 1.0, v_tilde_mat_column,
			   current_lanczos_vector);
      }

      // Compute l_1
      double l_1 = s_vec[q] * alpha - c_vec[q] * beta_tilde_vec[q];

      // Compute l_2
      double l_2 = s_vec[q] * beta_vec[q];
      
      // Compute alpha_tilde
      double alpha_tilde = -s_vec[q] * beta_tilde_vec[q] - c_vec[q] * alpha;
      
      // Compute beta_tilde
      beta_tilde_vec[q] = c_vec[q] * beta_vec[q];

      double l_0 = sqrt(alpha_tilde * alpha_tilde + beta_vec[q] * beta_vec[q]);

      if(l_0 != 0) {
	c_vec[q] = alpha_tilde / l_0;
	s_vec[q] = beta_vec[q] / l_0;
      }
      else {
	printf("Warning: Division by zero attempted!\n");
      }
      
      double g_tilde = g_double_tilde_vec[q] - l_1 * g_vec[q];
      g_double_tilde_vec[q] = -l_2 * g_vec[q];

      if(l_0 != 0) {
	g_vec[q] = g_tilde / l_0;
      }
      else {
	printf("Warning: Division by zero attempted!\n");
      }
      
      // Update solution.
      la::AddExpert(row_length_, g_vec[q] * c_vec[q], w_mat.GetColumnPtr(q),
		    solution_vectors_e_.GetColumnPtr(q));
      la::AddExpert(row_length_, g_vec[q] * s_vec[q], current_lanczos_vector,
		    solution_vectors_e_.GetColumnPtr(q));

      la::Scale(row_length_, s_vec[q], w_mat.GetColumnPtr(q));
      la::AddExpert(row_length_, -c_vec[q], current_lanczos_vector,
		    w_mat.GetColumnPtr(q));
      
    } // end of iterating over each query point.
    
  } // end of an iteration of SYMMLQ
}
