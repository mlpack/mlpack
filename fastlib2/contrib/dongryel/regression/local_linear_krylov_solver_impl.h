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
void LocalLinearKrylov<TKernel>::DualtreeSolverCanonical_
(Tree *qnode, Tree *rnode, Matrix &current_lanczos_vectors,
 DRange &root_negative_dot_product_range,
 DRange &root_positive_dot_product_range) {
  
  
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DotProductBetweenTwoBounds_
(Tree *qnode, Tree *rnode, DRange &negative_dot_product_range,
 DRange &positive_dot_product_range) {
  
  DHrectBound<2> lanczos_vectors_bound = qnode->stat().lanczos_vectors_bound_;

  // Initialize the dot-product ranges.
  negative_dot_product_range.lo = negative_dot_product_range.hi = 0;
  positive_dot_product_range.lo = positive_dot_product_range.hi = 0;

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
  (qnode->stat().postponed_ll_vector_l_).SetZero();
  (qnode->stat().postponed_ll_vector_e_).SetZero();
  (qnode->stat().postponed_ll_vector_u_).SetZero();

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
void LocalLinearKrylov<TKernel>::SolveLeastSquaresByKrylov_() {

  // Temporary variables needed for Lanczos iteration...
  Matrix previous_lanczos_vectors;
  Matrix current_lanczos_vectors;
  previous_lanczos_vectors.Init(row_length_, qset_.n_cols());
  current_lanczos_vectors.Init(row_length_, qset_.n_cols());
  Matrix omega;
  omega.Init(row_length_, qset_.n_cols());
  Vector alpha, beta, rho, previous_rho, lambda;
  Vector zeta, previous_zeta;
  alpha.Init(qset_.n_cols());
  beta.Init(qset_.n_cols());
  rho.Init(qset_.n_cols());
  previous_rho.Init(qset_.n_cols());
  lambda.Init(qset_.n_cols());
  zeta.Init(qset_.n_cols());
  previous_zeta.Init(qset_.n_cols());

  // Temporary variables to hold dot product ranges for the root nodes
  // of the two trees.
  DRange root_negative_dot_product_range, root_positive_dot_product_range;
  
  // Initialize the initial solutions to zero vectors.
  solution_vectors_e_.SetZero();

  // Initialize the query tree solution bounds.
  InitializeQueryTreeSolutionBound_(qroot_, current_lanczos_vectors);

  // Compute the dot product bounds.
  DotProductBetweenTwoBounds_(qroot_, rroot_, root_negative_dot_product_range,
			      root_positive_dot_product_range);

  // Initialize the query tree bound statistics.
  InitializeQueryTreeSumBound_(qroot_, root_negative_dot_product_range,
			       root_positive_dot_product_range);
  
  // Main iteration of the Lanczos - repeat until "convergence"...
  for(index_t m = 0; m < row_length_; m++) {

    // Multiply the current lanczos vector with the linear operator.
    DualtreeSolverCanonical_(qroot_, rroot_, current_lanczos_vectors,
			     root_negative_dot_product_range,
			     root_positive_dot_product_range);
    
    // Take the dot product between the product above and the current
    // lanczos vector.

    if(m > 0) {
      
    }
  }
}
