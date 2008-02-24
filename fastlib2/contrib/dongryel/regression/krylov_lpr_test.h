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
      double kernel_value = kernels_[r].EvalUnnormOnSq(dsqd);

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
