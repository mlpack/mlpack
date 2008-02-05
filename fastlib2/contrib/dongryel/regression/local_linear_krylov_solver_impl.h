// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void LocalLinearKrylov<TKernel>::DotProductBetweenTwoBounds_
(Tree *qnode, Tree *rnode, DRange &negative_dot_product_range,
 DRange &positive_dot_product_range) {
  
  DHrectBound<2> bound_for_solutions = qnode->stat().bound_for_solutions_;

  // Initialize the dot-product ranges.
  negative_dot_product_range.lo = negative_dot_product_range.hi = 0;
  positive_dot_product_range.lo = positive_dot_product_range.hi = 0;

  for(index_t d = 1; d <= dimension_; d++) {

    const DRange &solution_directional_bound = bound_for_solutions.get(d);
    const DRange &reference_node_directional_bound = rnode->bound().get(d - 1);
    
    if(solution_directional_bound.lo > 0) {
      positive_dot_product_range.hi += solution_directional_bound.hi *
	reference_node_directional_bound.hi;
    }
    else if(solution_directional_bound.Contains(0)) {
      positive_dot_product_range.hi += solution_directional_bound.hi *
	reference_node_directional_bound.hi;
      negative_dot_product_range.lo += solution_directional_bound.lo *
	reference_node_directional_bound.hi;
    }
    else {
      negative_dot_product_range.lo += solution_directional_bound.lo *
	reference_node_directional_bound.hi;
      negative_dot_product_range.hi += solution_directional_bound.hi *
	reference_node_directional_bound.lo;
    }    
  } // End of looping over each component...
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::InitializeQueryTreeSolutionBound_
(Tree *qnode) {

  // If the query node is a leaf, then exhaustively iterate over and
  // form bounding boxes of the current solution.
  if(qnode->is_leaf()) {
    (qnode->stat().bound_for_solutions_).Reset();
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      Vector solution_vector;
      solution_vectors_e_.MakeColumnVector(q, &solution_vector);      
      qnode->stat().bound_for_solutions_ |= solution_vector;
    }
  }

  // Otherwise, traverse the left and the right and combine the
  // bounding boxes of the solutions for the two children.
  else {    
    InitializeQueryTreeSolutionBound_(qnode->left());
    InitializeQueryTreeSolutionBound_(qnode->right());
    
    (qnode->stat().bound_for_solutions_).Reset();
    qnode->stat().bound_for_solutions_ |= 
      (qnode->left()->stat()).bound_for_solutions_;
    qnode->stat().bound_for_solutions_ |=
      (qnode->right()->stat()).bound_for_solutions_;
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

  if(!qnode->is_leaf()) {
    
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

  // Temporary variables to hold dot product ranges for the root nodes
  // of the two trees.
  DRange root_negative_dot_product_range, root_positive_dot_product_range;
  
  // Initialize the initial solutions to zero vectors.
  solution_vectors_e_.SetZero();

  // Initialize the query tree solution bounds.
  InitializeQueryTreeSolutionBound_(qroot_);

  // Compute the dot product bounds.
  DotProductBetweenTwoBounds_(qroot_, rroot_, root_negative_dot_product_range,
			      root_positive_dot_product_range);

  // Initialize the query tree bound statistics.
  InitializeQueryTreeSumBound_(qroot_, root_negative_dot_product_range,
			       root_positive_dot_product_range);

}
