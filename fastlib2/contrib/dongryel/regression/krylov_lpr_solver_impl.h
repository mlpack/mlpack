// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::DualtreeSolverBase_
(QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
 const ArrayList<bool> &query_should_exit_the_loop,
 const Matrix &current_lanczos_vectors,
 Matrix &lanczos_prod_l, Matrix &lanczos_prod_e,
 Vector &lanczos_prod_used_error, Vector &lanczos_prod_n_pruned,
 Matrix &neg_lanczos_prod_e, Matrix &neg_lanczos_prod_u,
 Vector &neg_lanczos_prod_used_error, Vector &neg_lanczos_prod_n_pruned) {

  // Temporary variable for storing multivariate expansion of a
  // reference point.
  Vector reference_point_expansion;
  reference_point_expansion.Init(row_length_);
  
  // Clear the summary statistics of the current query node so that we
  // can refine it to better bounds.
  qnode->stat().ll_vector_norm_l_ = DBL_MAX;
  qnode->stat().ll_vector_used_error_ = 0;
  qnode->stat().ll_vector_n_pruned_ = DBL_MAX;
  qnode->stat().neg_ll_vector_norm_l_ = DBL_MAX;
  qnode->stat().neg_ll_vector_used_error_ = 0;
  qnode->stat().neg_ll_vector_n_pruned_ = DBL_MAX;
  
  // for each query point
  for(index_t q = qnode->begin(); q < qnode->end(); q++) {

    // This is potentially inefficient and could be solved by
    // rebuilding the query tree everytime when a query point exists
    // the Lanczos outer loop.
    if(query_should_exit_the_loop[q]) {
      continue;
    }

    // get query point.
    const double *q_col = qset.GetColumnPtr(q);

    // Get the query point's associated current Lanczos vector.
    const double *q_lanczos_vector = current_lanczos_vectors.GetColumnPtr(q);

    // get the column vectors accumulating the sums to update.
    double *q_lanczos_prod_l = lanczos_prod_l.GetColumnPtr(q);
    double *q_lanczos_prod_e = lanczos_prod_e.GetColumnPtr(q);
    double *q_neg_lanczos_prod_e = neg_lanczos_prod_e.GetColumnPtr(q);
    double *q_neg_lanczos_prod_u = neg_lanczos_prod_u.GetColumnPtr(q);

    // Incorporate the postponed information.
    la::AddTo(row_length_, (qnode->stat().postponed_ll_vector_l_).ptr(),
	      q_lanczos_prod_l);
    lanczos_prod_used_error[q] += 
      qnode->stat().postponed_ll_vector_used_error_;
    lanczos_prod_n_pruned[q] += qnode->stat().postponed_ll_vector_n_pruned_;

    la::AddTo(row_length_, (qnode->stat().postponed_neg_ll_vector_u_).ptr(),
	      q_neg_lanczos_prod_u);
    neg_lanczos_prod_used_error[q] +=
      qnode->stat().postponed_neg_ll_vector_used_error_;
    neg_lanczos_prod_n_pruned[q] += 
      qnode->stat().postponed_neg_ll_vector_n_pruned_;
    
    // for each reference point
    for(index_t r = rnode->begin(); r < rnode->end(); r++) {
      
      // get reference point.
      const double *r_col = rset_.GetColumnPtr(r);

      // Compute the reference point expansion.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, r_col, reference_point_expansion.ptr());
      
      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[r].EvalUnnormOnSq(dsqd);

      // Take the dot product between the query point's Lanczos vector
      // and the reference point expansion.
      double dot_product = la::Dot(row_length_, q_lanczos_vector, 
				   reference_point_expansion.ptr());
      double front_factor = dot_product * kernel_value;

      // For each vector component, update the lower/estimate/upper
      // bound quantities.
      if(front_factor > 0) {
	la::AddExpert(row_length_, front_factor, 
		      reference_point_expansion.ptr(), q_lanczos_prod_l);
	la::AddExpert(row_length_, front_factor,
		      reference_point_expansion.ptr(), q_lanczos_prod_e);
      }
      else {
	la::AddExpert(row_length_, front_factor,
		      reference_point_expansion.ptr(), q_neg_lanczos_prod_e);
	la::AddExpert(row_length_, front_factor,
		      reference_point_expansion.ptr(), q_neg_lanczos_prod_u);
      }

    } // end of iterating over each reference point.

    // Update the pruned quantities.
    lanczos_prod_n_pruned[q] += 
      rnode->stat().sum_reference_point_expansion_norm_;
    neg_lanczos_prod_n_pruned[q] += 
      rnode->stat().sum_reference_point_expansion_norm_;

    // Now, loop over each vector component for the current query and
    // correct the upper bound by the assumption made in the
    // initialization phase of the query tree. Refine min and max
    // summary statistics.
    qnode->stat().ll_vector_norm_l_ =
      std::min(qnode->stat().ll_vector_norm_l_,
	       MatrixUtil::EntrywiseLpNorm(row_length_, q_lanczos_prod_l, 1));
    qnode->stat().ll_vector_used_error_ =
      std::max(qnode->stat().ll_vector_used_error_,
	       lanczos_prod_used_error[q]);
    qnode->stat().ll_vector_n_pruned_ =
      std::min(qnode->stat().ll_vector_n_pruned_, lanczos_prod_n_pruned[q]);
    
    qnode->stat().neg_ll_vector_norm_l_ =
      std::min(qnode->stat().neg_ll_vector_norm_l_,
	       MatrixUtil::EntrywiseLpNorm(row_length_, 
					   q_neg_lanczos_prod_u, 1));
    qnode->stat().neg_ll_vector_used_error_ =
      std::max(qnode->stat().neg_ll_vector_used_error_,
	       neg_lanczos_prod_used_error[q]);
    qnode->stat().neg_ll_vector_n_pruned_ =
      std::min(qnode->stat().neg_ll_vector_n_pruned_, 
	       neg_lanczos_prod_n_pruned[q]);

  } // end of iterating over each query point.

  // Clear postponed information.
  (qnode->stat().postponed_ll_vector_l_).SetZero();
  qnode->stat().postponed_ll_vector_used_error_ = 0;
  qnode->stat().postponed_ll_vector_n_pruned_ = 0;
  (qnode->stat().postponed_neg_ll_vector_u_).SetZero();
  qnode->stat().postponed_neg_ll_vector_used_error_ = 0;
  qnode->stat().postponed_neg_ll_vector_n_pruned_ = 0;
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::DualtreeSolverCanonical_
(QueryTree *qnode, ReferenceTree *rnode, const Matrix &qset,
 const ArrayList<bool> &query_should_exit_the_loop,
 const Matrix &current_lanczos_vectors, 
 Matrix &lanczos_prod_l, Matrix &lanczos_prod_e,
 Vector &lanczos_prod_used_error, Vector &lanczos_prod_n_pruned,
 Matrix &neg_lanczos_prod_e, Matrix &neg_lanczos_prod_u,
 Vector &neg_lanczos_prod_used_error, Vector &neg_lanczos_prod_n_pruned) {
    
  // Variables for storing changes due to a prune.
  double delta_used_error = 0, delta_n_pruned = 0, delta_neg_used_error = 0, 
    delta_neg_n_pruned = 0;
  Vector delta_l, delta_e, delta_neg_u, delta_neg_e;
  delta_l.Init(row_length_);
  delta_e.Init(row_length_);
  delta_neg_u.Init(row_length_);
  delta_neg_e.Init(row_length_);

  // temporary variable for holding distance/kernel value bounds
  DRange dsqd_range;
  DRange kernel_value_range;

  // First compute distance/kernel value bounds and dot product bound
  // ranges.
  LprUtil::SqdistAndKernelRanges_(qnode, rnode, dsqd_range, 
				  kernel_value_range);

  // Temporary variables hold the dot product ranges.
  DRange negative_dot_product_range, positive_dot_product_range;
  
  // Compute the dot product range.
  DotProductBetweenTwoBounds_(qnode, rnode, negative_dot_product_range,
			      positive_dot_product_range);

  // try finite difference pruning first
  if(TPruneRule::PrunableKrylovSolver
     (internal_relative_error_,
      rnode->stat().sum_reference_point_expansion_norm_,
      qnode, rnode, dsqd_range, kernel_value_range, 
      negative_dot_product_range, positive_dot_product_range,
      delta_l, delta_e, delta_used_error, delta_n_pruned,
      delta_neg_u, delta_neg_e, delta_neg_used_error, delta_neg_n_pruned)) {

    la::AddTo(delta_l, &(qnode->stat().postponed_ll_vector_l_));
    la::AddTo(delta_e, &(qnode->stat().postponed_ll_vector_e_));
    qnode->stat().postponed_ll_vector_used_error_ += delta_used_error;
    qnode->stat().postponed_ll_vector_n_pruned_ += delta_n_pruned;

    la::AddTo(delta_neg_u, &(qnode->stat().postponed_neg_ll_vector_u_));
    la::AddTo(delta_neg_e, &(qnode->stat().postponed_neg_ll_vector_e_));
    qnode->stat().postponed_neg_ll_vector_used_error_ += delta_neg_used_error;
    qnode->stat().postponed_neg_ll_vector_n_pruned_ += delta_neg_n_pruned;

    num_finite_difference_prunes_++;

    return;
  }
  
  // For the Epanechnikov kernel, we can prune using the far field
  // moments if the maximum distance between the two nodes is within
  // the bandwidth! This if-statement does not apply to the Gaussian
  // kernel, so I need to fix in the future!
  if(rnode->stat().min_bandwidth_kernel.bandwidth_sq() >= dsqd_range.hi && 
     rnode->count() > dimension_ * dimension_) {

    la::AddTo(delta_l, &(qnode->stat().postponed_ll_vector_l_));
    qnode->stat().postponed_ll_vector_n_pruned_ += delta_n_pruned;

    la::AddTo(delta_neg_u, &(qnode->stat().postponed_neg_ll_vector_u_));
    qnode->stat().postponed_neg_ll_vector_n_pruned_ += delta_neg_n_pruned;

    // Add the Epanechnikov moments
    for(index_t j = 0; j < row_length_; j++) {
      for(index_t i = 0; i <= j; i++) {
	qnode->stat().postponed_epanechnikov_moments_[j][i].
	  Add(rnode->stat().data_outer_products_far_field_expansion_[j][i]);
      }
    }

    // Keep track of the far-field prunes.
    num_epanechnikov_prunes_++;
    return;
  }
  
  // for leaf query node
  if(qnode->is_leaf()) {
    
    // for leaf pairs, go exhaustive
    if(rnode->is_leaf()) {
      DualtreeSolverBase_
	(qnode, rnode, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
      return;
    }
    
    // for non-leaf reference, expand reference node
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      LprUtil::BestReferenceNodePartners(qnode, rnode->left(), rnode->right(), 
					 &rnode_first, &rnode_second);
      DualtreeSolverCanonical_
	(qnode, rnode_first, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
      DualtreeSolverCanonical_
	(qnode, rnode_second, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);

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
    // node to the children of the query node.
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

    la::AddTo(q_stat.postponed_neg_ll_vector_u_,
	      &(q_left_stat.postponed_neg_ll_vector_u_));
    la::AddTo(q_stat.postponed_neg_ll_vector_u_,
	      &(q_right_stat.postponed_neg_ll_vector_u_));
    q_left_stat.postponed_neg_ll_vector_used_error_ +=
      q_stat.postponed_neg_ll_vector_used_error_;
    q_right_stat.postponed_neg_ll_vector_used_error_ +=
      q_stat.postponed_neg_ll_vector_used_error_;
    q_left_stat.postponed_neg_ll_vector_n_pruned_ +=
      q_stat.postponed_neg_ll_vector_n_pruned_;
    q_right_stat.postponed_neg_ll_vector_n_pruned_ +=
      q_stat.postponed_neg_ll_vector_n_pruned_;    

    // Clear the statistics after pushing them downwards.
    q_stat.postponed_ll_vector_l_.SetZero();
    q_stat.postponed_ll_vector_used_error_ = 0;
    q_stat.postponed_ll_vector_n_pruned_ = 0;
    q_stat.postponed_neg_ll_vector_u_.SetZero();
    q_stat.postponed_neg_ll_vector_used_error_ = 0;
    q_stat.postponed_neg_ll_vector_n_pruned_ = 0;

    // For a leaf reference node, expand query node
    if(rnode->is_leaf()) {
      QueryTree *qnode_first = NULL, *qnode_second = NULL;
      
      LprUtil::BestQueryNodePartners(rnode, qnode->left(), qnode->right(), 
				     &qnode_first, &qnode_second);
      DualtreeSolverCanonical_
	(qnode_first, rnode, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
      DualtreeSolverCanonical_
	(qnode_second, rnode, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
    }
    
    // for non-leaf reference node, expand both query and reference nodes
    else {
      ReferenceTree *rnode_first = NULL, *rnode_second = NULL;
      
      LprUtil::BestReferenceNodePartners(qnode->left(), rnode->left(), 
					 rnode->right(), &rnode_first, 
					 &rnode_second);
      DualtreeSolverCanonical_
	(qnode->left(), rnode_first, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
      DualtreeSolverCanonical_
	(qnode->left(), rnode_second, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
      
      LprUtil::BestReferenceNodePartners(qnode->right(), rnode->left(), 
					 rnode->right(), &rnode_first, 
					 &rnode_second);
      DualtreeSolverCanonical_
	(qnode->right(), rnode_first, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);

      DualtreeSolverCanonical_
	(qnode->right(), rnode_second, qset, query_should_exit_the_loop,
	 current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
	 lanczos_prod_used_error, lanczos_prod_n_pruned,
	 neg_lanczos_prod_e, neg_lanczos_prod_u,
	 neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
    }
    
    // reaccumulate the summary statistics.
    q_stat.ll_vector_norm_l_ =
      std::min
      (q_left_stat.ll_vector_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_left_stat.postponed_ll_vector_l_, 1),
       q_right_stat.ll_vector_norm_l_ +
       MatrixUtil::EntrywiseLpNorm(q_right_stat.postponed_ll_vector_l_, 1));
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

    q_stat.neg_ll_vector_norm_l_ =
      std::min(q_left_stat.neg_ll_vector_norm_l_ +
	       MatrixUtil::EntrywiseLpNorm
	       (q_left_stat.postponed_neg_ll_vector_u_, 1),
	       q_right_stat.neg_ll_vector_norm_l_ +
	       MatrixUtil::EntrywiseLpNorm
	       (q_right_stat.postponed_neg_ll_vector_u_, 1));
    q_stat.neg_ll_vector_used_error_ =
      std::max(q_left_stat.neg_ll_vector_used_error_ +
	       q_left_stat.postponed_neg_ll_vector_used_error_,
	       q_right_stat.neg_ll_vector_used_error_ +
	       q_right_stat.postponed_neg_ll_vector_used_error_);
    q_stat.neg_ll_vector_n_pruned_ =
      std::min(q_left_stat.neg_ll_vector_n_pruned_ +
	       q_left_stat.postponed_neg_ll_vector_n_pruned_,
	       q_right_stat.neg_ll_vector_n_pruned_ +
	       q_right_stat.postponed_neg_ll_vector_n_pruned_);
    return;
  } // end of the case: non-leaf query node.
  
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::DotProductBetweenTwoBounds_
(QueryTree *qnode, ReferenceTree *rnode, DRange &negative_dot_product_range,
 DRange &positive_dot_product_range) {
  
  DHrectBound<2> lanczos_vectors_bound = qnode->stat().lanczos_vectors_bound_;

  // Initialize the dot-product ranges.
  negative_dot_product_range.lo = negative_dot_product_range.hi = 0;
  positive_dot_product_range.lo = positive_dot_product_range.hi = 0;

  for(index_t d = 0; d < row_length_; d++) {

    const DRange &lanczos_directional_bound = lanczos_vectors_bound.get(d);
    const DRange &reference_node_directional_bound = 
      rnode->stat().reference_point_expansion_bound_.get(d);
    
    if(lanczos_directional_bound.lo > 0) {
      positive_dot_product_range.lo += lanczos_directional_bound.lo *
	reference_node_directional_bound.lo;
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

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::InitializeQueryTreeLanczosVectorBound_
(QueryTree *qnode, const Matrix &qset, 
 const ArrayList<bool> &exclude_query_flag,
 const Matrix &current_lanczos_vectors) {

  // Set the bound quantities to default.
  qnode->stat().Reset();

  // If the query node is a leaf, then exhaustively iterate over and
  // form bounding boxes of the current solution.
  if(qnode->is_leaf()) {
    qnode->bound().Reset();

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // If the current query point is not to be included in the
      // bounding box, then skip it.
      if(exclude_query_flag[q]) {
	continue;
      }

      Vector query_vector;
      Vector lanczos_vector;
      current_lanczos_vectors.MakeColumnVector(q, &lanczos_vector);
      qset.MakeColumnVector(q, &query_vector);
      qnode->stat().lanczos_vectors_bound_ |= lanczos_vector;
      qnode->bound() |= query_vector;
    }
  }

  // Otherwise, traverse the left and the right and combine the
  // bounding boxes of the solutions for the two children.
  else {    
    InitializeQueryTreeLanczosVectorBound_(qnode->left(), qset,
					   exclude_query_flag,
					   current_lanczos_vectors);
    InitializeQueryTreeLanczosVectorBound_(qnode->right(), qset,
					   exclude_query_flag,
					   current_lanczos_vectors);
    
    // Reset the bounding box for the Lanczos vectors and reform it
    // using the bounding boxes owned by the children.
    qnode->stat().lanczos_vectors_bound_ |= 
      (qnode->left()->stat()).lanczos_vectors_bound_;
    qnode->stat().lanczos_vectors_bound_ |=
      (qnode->right()->stat()).lanczos_vectors_bound_;

    // Ditto for the bounding box for the query points.
    qnode->bound().Reset();
    qnode->bound() |= qnode->left()->bound();
    qnode->bound() |= qnode->right()->bound();
  }
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::FinalizeQueryTreeLanczosMultiplier_
(QueryTree *qnode, const Matrix &qset,
 const ArrayList<bool> &exclude_query_flag,
 const Matrix &current_lanczos_vectors,
 Matrix &lanczos_prod_l, Matrix &lanczos_prod_e,
 Vector &lanczos_prod_used_error, Vector &lanczos_prod_n_pruned,
 Matrix &neg_lanczos_prod_e, Matrix &neg_lanczos_prod_u,
 Vector &neg_lanczos_prod_used_error, Vector &neg_lanczos_prod_n_pruned) {

  KrylovLprQStat<TKernel> &q_stat = qnode->stat();

  if(qnode->is_leaf()) {
    
    // The matrix to store the evaluated moments at each query point.
    Matrix evaluated_moments;
    evaluated_moments.Init(row_length_, row_length_);
    Vector evaluated_moments_times_lanczos_vector;
    evaluated_moments_times_lanczos_vector.Init(row_length_);

    // Iterate over each query point.
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      if(exclude_query_flag[q]) {
	continue;
      }

      // Get the current query point.
      Vector q_col;
      qset.MakeColumnVector(q, &q_col);

      // Get the pointer to the current lanczos vector owned by the
      // current query point.
      Vector q_current_lanczos_vector;
      current_lanczos_vectors.MakeColumnVector(q, &q_current_lanczos_vector);

      // Get the column vectors accumulating the sums to update.
      double *q_lanczos_prod_l = lanczos_prod_l.GetColumnPtr(q);
      double *q_lanczos_prod_e = lanczos_prod_e.GetColumnPtr(q);
      double *q_neg_lanczos_prod_e = neg_lanczos_prod_e.GetColumnPtr(q);
      double *q_neg_lanczos_prod_u = neg_lanczos_prod_u.GetColumnPtr(q);
      
      // Incorporate the postponed information.
      la::AddTo(row_length_, (q_stat.postponed_ll_vector_l_).ptr(),
		q_lanczos_prod_l);
      la::AddTo(row_length_, (q_stat.postponed_ll_vector_e_).ptr(),
		q_lanczos_prod_e);
      la::AddTo(row_length_, (q_stat.postponed_neg_ll_vector_e_).ptr(),
		q_neg_lanczos_prod_e);
      la::AddTo(row_length_, (q_stat.postponed_neg_ll_vector_u_).ptr(),
		q_neg_lanczos_prod_u);

      // Evaluate the Epanechnikov moments.
      for(index_t i = 0; i < row_length_; i++) {
	for(index_t j = 0; j <= i; j++) {
	  evaluated_moments.set
	    (j, i, qnode->stat().postponed_epanechnikov_moments_[i][j].
	     ComputeKernelSum(q_col));
	}
      }
      for(index_t i = 0; i < row_length_; i++) {
	for(index_t j = i + 1; j < row_length_; j++) {
	  evaluated_moments.set(j, i, evaluated_moments.get(i, j));
	}
      }

      // Now compute the product between the evaluated moments and the
      // Lanczos vector owned by this query point.
      la::MulOverwrite(evaluated_moments, q_current_lanczos_vector,
		       &evaluated_moments_times_lanczos_vector);

      // Now accumulate the sum depending on the negativity or the
      // positivity of each component.
      for(index_t i = 0; i < row_length_; i++) {
	if(evaluated_moments_times_lanczos_vector[i] > 0) {
	  q_lanczos_prod_e[i] += evaluated_moments_times_lanczos_vector[i];
	}
	else {
	  q_neg_lanczos_prod_e[i] += evaluated_moments_times_lanczos_vector[i];
	}
      }

    } // end of iterating over each query point.
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

    la::AddTo(q_stat.postponed_neg_ll_vector_e_,
              &(q_left_stat.postponed_neg_ll_vector_e_));
    la::AddTo(q_stat.postponed_neg_ll_vector_e_,
              &(q_right_stat.postponed_neg_ll_vector_e_));
    la::AddTo(q_stat.postponed_neg_ll_vector_u_,
              &(q_left_stat.postponed_neg_ll_vector_u_));
    la::AddTo(q_stat.postponed_neg_ll_vector_u_,
              &(q_right_stat.postponed_neg_ll_vector_u_));

    // Push down Epanechnikov pruned portions.
    for(index_t i = 0; i < row_length_; i++) {
      for(index_t j = 0; j <= i; j++) {
	q_left_stat.postponed_epanechnikov_moments_[i][j].Add
	  (q_stat.postponed_epanechnikov_moments_[i][j]);
	q_right_stat.postponed_epanechnikov_moments_[i][j].Add
	  (q_stat.postponed_epanechnikov_moments_[i][j]);
      }
    }

    // Recurse both branches of the query node.
    FinalizeQueryTreeLanczosMultiplier_
      (qnode->left(), qset, exclude_query_flag, current_lanczos_vectors,
       lanczos_prod_l, lanczos_prod_e, lanczos_prod_used_error, 
       lanczos_prod_n_pruned, neg_lanczos_prod_e, neg_lanczos_prod_u,
       neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
    FinalizeQueryTreeLanczosMultiplier_
      (qnode->right(), qset, exclude_query_flag, current_lanczos_vectors,
       lanczos_prod_l, lanczos_prod_e, lanczos_prod_used_error,
       lanczos_prod_n_pruned, neg_lanczos_prod_e, neg_lanczos_prod_u,
       neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);
  }
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::SolveLeastSquaresByKrylov_
(QueryTree *qroot, const Matrix &qset, const Matrix &right_hand_sides,
 Matrix &solution_vectors_e) {

  // Initialize the initial solutions to be zero vectors.
  solution_vectors_e.SetZero();

  // Temporary variables needed for SYMMLQ iteration...
  Matrix previous_lanczos_vectors;
  Matrix current_lanczos_vectors;
  Matrix v_tilde_mat;
  previous_lanczos_vectors.Init(row_length_, qset.n_cols()); 
  v_tilde_mat.Init(row_length_, qset.n_cols());

  Vector g_double_tilde_vec;
  Vector g_vec;
  g_double_tilde_vec.Init(qset.n_cols());
  g_vec.Init(qset.n_cols());
  
  current_lanczos_vectors.Init(row_length_, qset.n_cols());

  // More temporary variables for SYMMLQ routine...
  Vector c_vec, beta_vec, beta_tilde_vec, s_vec;
  Matrix w_mat;
  c_vec.Init(qset.n_cols());
  beta_vec.Init(qset.n_cols());
  beta_tilde_vec.Init(qset.n_cols());
  s_vec.Init(qset.n_cols());
  w_mat.Init(row_length_, qset.n_cols());
  
  // Initialize before entering the main iteration... This
  // initialization implicitly assumes that initial guess to the
  // linear system is the zero vector.
  current_lanczos_vectors.CopyValues(right_hand_sides);
  NormalizeMatrixColumnVectors_(current_lanczos_vectors, g_double_tilde_vec);
  beta_vec.SetZero();
  beta_tilde_vec.SetZero();
  c_vec.SetAll(-1);
  s_vec.SetZero();
  previous_lanczos_vectors.SetZero();
  w_mat.CopyValues(current_lanczos_vectors);
  g_vec.SetZero();

  // Flag to tell whether each query stays in the Krylov loop or not.
  ArrayList<bool> query_should_exit_the_loop;
  query_should_exit_the_loop.Init(qset.n_cols());

  // Set the boolean flags to false
  for(index_t q = 0; q < qset.n_cols(); q++) {
    query_should_exit_the_loop[q] = false;
  }

  // Initialize variables necessary for the dual-tree computation...
  Matrix lanczos_prod_l, lanczos_prod_e, neg_lanczos_prod_e, 
    neg_lanczos_prod_u;
  Vector lanczos_prod_used_error, lanczos_prod_n_pruned,
    neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned;
  lanczos_prod_l.Init(row_length_, qset.n_cols());
  lanczos_prod_e.Init(row_length_, qset.n_cols());
  neg_lanczos_prod_e.Init(row_length_, qset.n_cols());
  neg_lanczos_prod_u.Init(row_length_, qset.n_cols());
  lanczos_prod_used_error.Init(qset.n_cols());
  lanczos_prod_n_pruned.Init(qset.n_cols());
  neg_lanczos_prod_used_error.Init(qset.n_cols());
  neg_lanczos_prod_n_pruned.Init(qset.n_cols());

  // Main iteration of the SYMMLQ algorithm - repeat until
  // "convergence"...
  for(index_t num_iter = 0; num_iter < sqrt(row_length_); num_iter++) {

    // Determine how many queries are in the Krylov loop.
    int num_queries_in_krylov_loop = 0;
    for(index_t q = 0; q < qset.n_cols(); q++) {
      if(!query_should_exit_the_loop[q]) {
	num_queries_in_krylov_loop++;
      }
    }
    if(num_queries_in_krylov_loop == 0) {
      break;
    }
    
    // Initialize the query tree Lanzcos vector bounds.
    InitializeQueryTreeLanczosVectorBound_
      (qroot, qset, query_should_exit_the_loop, current_lanczos_vectors);

    // Initialize the query tree bound statistics.
    lanczos_prod_l.SetZero();
    lanczos_prod_e.SetZero();
    lanczos_prod_used_error.SetZero();
    lanczos_prod_n_pruned.SetZero();
    neg_lanczos_prod_e.SetZero();
    neg_lanczos_prod_u.SetZero();
    neg_lanczos_prod_used_error.SetZero();
    neg_lanczos_prod_n_pruned.SetZero();

    // Multiply the current lanczos vector with the linear operator.
    DualtreeSolverCanonical_
      (qroot, rroot_, qset, query_should_exit_the_loop,
       current_lanczos_vectors, lanczos_prod_l, lanczos_prod_e,
       lanczos_prod_used_error, lanczos_prod_n_pruned, neg_lanczos_prod_e,
       neg_lanczos_prod_u, neg_lanczos_prod_used_error, 
       neg_lanczos_prod_n_pruned);
    FinalizeQueryTreeLanczosMultiplier_
      (qroot, qset, query_should_exit_the_loop, current_lanczos_vectors,
       lanczos_prod_l, lanczos_prod_e, lanczos_prod_used_error,
       lanczos_prod_n_pruned, neg_lanczos_prod_e, neg_lanczos_prod_u,
       neg_lanczos_prod_used_error, neg_lanczos_prod_n_pruned);

    // Compute v_tilde_mat (the residue after applying the linear
    // operator the current Lanczos vector).
    la::AddOverwrite(lanczos_prod_e, neg_lanczos_prod_e, &v_tilde_mat);
    
    /*
    printf("Positive matrix: %g\n",
	   MatrixUtil::EntrywiseLpNorm(lanczos_prod_e, 1));
    printf("Negative matrix: %g\n",
	   MatrixUtil::EntrywiseLpNorm(neg_lanczos_prod_e, 1));
    TestKrylovComputation_(qset, v_tilde_mat, current_lanczos_vectors,
			   query_should_exit_the_loop);
    */

    for(index_t q = 0; q < qset.n_cols(); q++) {

      // If the current query is not in the Krylov loop, skip it.
      if(query_should_exit_the_loop[q]) {
	continue;
      }

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
      for(index_t i = 0; i < row_length_; i++) {
	previous_lanczos_vector[i] = current_lanczos_vector[i];
      }

      // Set a new current Lanczos vector based on v_tilde_mat_column.
      // A potential place to watch out for division by zero!!
      if(beta_vec[q] > 0) {
	la::ScaleOverwrite(row_length_, 1.0 / beta_vec[q], v_tilde_mat_column,
			   current_lanczos_vector);
      }
      else {
	query_should_exit_the_loop[q] = true;
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

      // Another potential place to watch for division by zero!!
      if(l_0 != 0) {
	c_vec[q] = alpha_tilde / l_0;
	s_vec[q] = beta_vec[q] / l_0;
      }
      else {
	query_should_exit_the_loop[q] = true;
      }
      
      double g_tilde = g_double_tilde_vec[q] - l_1 * g_vec[q];
      g_double_tilde_vec[q] = -l_2 * g_vec[q];

      // Another potential place to watch for division by zero!!
      if(l_0 != 0) {
	g_vec[q] = g_tilde / l_0;
      }
      else {
	query_should_exit_the_loop[q] = true;
      }
      
      // Update solution...
      {
	la::AddExpert(row_length_, g_vec[q] * c_vec[q], w_mat.GetColumnPtr(q),
		      solution_vectors_e.GetColumnPtr(q));
	la::AddExpert(row_length_, g_vec[q] * s_vec[q], current_lanczos_vector,
		      solution_vectors_e.GetColumnPtr(q));
	
	la::Scale(row_length_, s_vec[q], w_mat.GetColumnPtr(q));
	la::AddExpert(row_length_, -c_vec[q], current_lanczos_vector,
		      w_mat.GetColumnPtr(q));
      }

      // Another criterion for quitting the Krylov loop...
      if(sqrt(g_tilde * g_tilde + g_double_tilde_vec[q] * 
	      g_double_tilde_vec[q]) < 0.001) {
	query_should_exit_the_loop[q] = true;
      }

    } // end of iterating over each query point.

  } // end of an iteration of SYMMLQ
}
