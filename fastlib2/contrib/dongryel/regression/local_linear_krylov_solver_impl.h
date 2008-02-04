// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
double LocalLinearKrylov<TKernel>::MaxDotProductBetweenTwoBounds_
(Tree *qnode, Tree *rnode) {
  
  DHrectBound<2> bound_for_solutions = qnode->stat().bound_for_solutions_;
  double max_dot_product = bound_for_solutions.get(0).hi;

  for(index_t d = 1; d <= dimension_; d++) {
    DRange &solution_directional_bound = bound_for_solutions.get(d);
    DRange &reference_node_directional_bound = rnode->bound().get(d - 1);

    double prod_solution_min_reference_min = 
      solution_directional_bound.lo * reference_node_directional_bound.lo;
    double prod_solution_min_reference_max = 
      solution_directional_bound.lo * reference_node_directional_bound.hi;
    double prod_solution_max_reference_min = 
      solution_directional_bound.hi * reference_node_directional_bound.lo;
    double prod_solution_max_reference_max = 
      solution_directional_bound.hi * reference_node_directional_bound.hi;

    max_dot_product += 
      std::max(prod_solution_min_reference_min,
	       std::max(prod_solution_min_reference_max,
			std::max(prod_solution_max_reference_min,
				 prod_solution_max_reference_max)));
  }
  return max_dot_product;
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::InitializeQueryTreeSolver_(Tree *qnode) {
  
  // Set the bounds to default values.
  (qnode->stat().ll_vector_l_).SetZero();
  (qnode->stat().ll_vector_u_).CopyValues
    (rroot_->stat().sum_targets_weighted_by_data_);
  (qnode->stat().postponed_ll_vector_l_).SetZero();
  (qnode->stat().postponed_ll_vector_e_).SetZero();
  (qnode->stat().postponed_ll_vector_u_).SetZero();

  // If the query node is a leaf, then exhaustively iterate over and
  // form bounding boxes of the current solution.
  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
    }
  }

  // Otherwise, traverse the left and the right and combine the
  // bounding boxes of the solutions for the two children.
  else {
    InitializeQueryTreeSolver_(qnode->left());
    InitializeQueryTreeSolver_(qnode->right());
    
  }
}

template<typename TKernel>
void LocalLinearKrylov<TKernel>::SolveLeastSquaresByKrylov_() {
  
  // Initialize the query tree bounds.
  InitializeQueryTreeSolver_(qroot_);
  
}
