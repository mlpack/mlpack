// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::TestRightHandSideComputation_
(const Matrix &qset, const Matrix &approximated) {
  
  Matrix exact_vector_e;
  exact_vector_e.Init(approximated.n_rows(), approximated.n_cols());
  exact_vector_e.SetZero();
  double max_relative_error = 0;

  for(index_t q = 0; q < qset.n_cols(); q++) {
    
    // get the column vector corresponding to the current query point.
    const double *q_col = qset.GetColumnPtr(q);

    // get the column vector accumulating the sum.
    Vector exact_vector_e_column, approx_column;
    exact_vector_e.MakeColumnVector(q, &exact_vector_e_column);
    approximated.MakeColumnVector(q, &approx_column);

    for(index_t r = 0; r < rset_.n_cols(); r++) {
    
      // get the column vector corresponding to the current reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // get the column vector containing the appropriate weights.
      const double *r_weights = target_weighted_rset_.GetColumnPtr(r);
      
      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[0].EvalUnnormOnSq(dsqd);

      // Add up the contribution of the reference point.
      la::AddExpert(row_length_, kernel_value, r_weights,
		    exact_vector_e_column.ptr());

    } // end of iterating over each reference point.

    double relative_error = 
      MatrixUtil::EntrywiseNormDifferenceRelative
      (exact_vector_e_column, approx_column, 1);

    max_relative_error = std::max(max_relative_error, relative_error);

  } // end of iterating over each query point.

  printf("Maximum relative error: %g\n", max_relative_error);
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::TestKrylovComputation_
(const Matrix &qset, const Matrix &approximated, 
 const Matrix &current_lanczos_vectors,
 const ArrayList<bool> &query_should_exit_the_loop) {
  
  double max_relative_error = 0;
  Matrix exact_vector_e;
  exact_vector_e.Init(approximated.n_rows(), approximated.n_cols());
  exact_vector_e.SetZero();

  Vector reference_point_expansion;
  reference_point_expansion.Init(row_length_);

  for(index_t q = 0; q < qset.n_cols(); q++) {
    
    // If the current query should not be computed, then skip it.
    if(query_should_exit_the_loop[q]) {
      continue;
    }

    // get the column vector corresponding to the current query point.
    const double *q_col = qset.GetColumnPtr(q);

    // get the column vector corresponding to the Lanczos vector owned
    // by the current query point.
    const double *q_lanczos_vector = current_lanczos_vectors.GetColumnPtr(q);

    // get the column vector accumulating the sum.
    Vector exact_vector_e_column, approx_column;
    exact_vector_e.MakeColumnVector(q, &exact_vector_e_column);
    approximated.MakeColumnVector(q, &approx_column);

    for(index_t r = 0; r < rset_.n_cols(); r++) {
    
      // get the column vector corresponding to the current reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[0].EvalUnnormOnSq(dsqd);

      // Compute the reference point expansion.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, r_col, reference_point_expansion.ptr());

      // Take the dot product between the query point's Lanczos vector
      // and [1 r^T]^T.
      double dot_product = 
	la::Dot(row_length_, q_lanczos_vector, 
		reference_point_expansion.ptr());
      double front_factor = dot_product * kernel_value;

      // Add the contribution of the current reference point.
      la::AddExpert(row_length_, front_factor, reference_point_expansion.ptr(),
		    exact_vector_e_column.ptr());

    } // end of iterating over each reference point.

    double relative_error = 
      MatrixUtil::EntrywiseNormDifferenceRelative
      (exact_vector_e_column, approx_column, 1);

    max_relative_error = std::max(max_relative_error, relative_error);

  } // end of iterating over each query point.

  printf("Maximum relative error: %g\n", max_relative_error);
}
