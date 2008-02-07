// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void LocalLinearKrylov<TKernel>::TestRightHandSideComputation_
(const Matrix &approximated) {
  
  Matrix exact_vector_e;
  exact_vector_e.Init(approximated.n_rows(), approximated.n_cols());
  exact_vector_e.SetZero();

  for(index_t q = 0; q < qset_.n_cols(); q++) {
    
    // get the column vector corresponding to the current query point.
    const double *q_col = qset_.GetColumnPtr(q);

    // get the column vector accumulating the sum.
    double *exact_vector_e_column = exact_vector_e.GetColumnPtr(q);

    for(index_t r = 0; r < rset_.n_cols(); r++) {
    
      // get the column vector corresponding to the current reference point.
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
	exact_vector_e_column[d] += kernel_value * r_weights[d];
      } // end of iterating over each vector component.

    } // end of iterating over each reference point.
      
  } // end of iterating over each query point.

  double max_relative_error = 0;
  
  for(index_t q = 0; q < qset_.n_cols(); q++) {

    // get the column vector containing the approximation.
    const double *approx_column = approximated.GetColumnPtr(q);

    // get the column vector accumulating the sum.
    double *exact_vector_e_column = exact_vector_e.GetColumnPtr(q);
    double l1_norm_exact_vector_e_column = 0;
    double l1_norm_diff_sum = 0;

    for(index_t d = 0; d <= dimension_; d++) {

      l1_norm_exact_vector_e_column += exact_vector_e_column[d];
      l1_norm_diff_sum += fabs(exact_vector_e_column[d] - approx_column[d]);
    }
    
    double relative_error = l1_norm_diff_sum / l1_norm_exact_vector_e_column;
    if(relative_error > max_relative_error) {
      max_relative_error = relative_error;
    }

  } // end of iterating over each query point.

  printf("Maximum relative error: %g\n", max_relative_error);
}
