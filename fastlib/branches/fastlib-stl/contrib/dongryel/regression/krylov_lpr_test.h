// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::TestDualtreeComputation_
(const Matrix &qset, const ArrayList<bool> *query_in_cg_loop,
 const bool confidence_band_computation_phase,
 const Vector &reference_weights, index_t column_index,
 const Matrix &approximated) {

  // temporary space for storing reference point expansion
  Vector r_col_expansion;
  r_col_expansion.Init(row_length_);

  Matrix exact_vector_e;
  exact_vector_e.Init(approximated.n_rows(), approximated.n_cols());
  exact_vector_e.SetZero();
  double max_relative_error = 0;

  for(index_t q = 0; q < qset.n_cols(); q++) {
    
    if(query_in_cg_loop != NULL && !((*query_in_cg_loop)[q])) {
      continue;
    }

    // get the column vector corresponding to the current query point.
    const double *q_col = qset.GetColumnPtr(q);

    // get the column vector accumulating the sum.
    Vector exact_vector_e_column, approx_column;
    exact_vector_e.MakeColumnVector(q, &exact_vector_e_column);
    approximated.MakeColumnVector(q, &approx_column);

    for(index_t r = 0; r < rset_.n_cols(); r++) {
    
      // get the column vector corresponding to the current reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // compute the reference point expansion
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, r_col, r_col_expansion.ptr());
      
      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[r].EvalUnnormOnSq(dsqd);
      
      if(confidence_band_computation_phase) {
	kernel_value *= kernel_value;
      }

      // Add up the contribution of the reference point.
      la::AddExpert(row_length_, kernel_value * reference_weights[r] *
		    r_col_expansion[column_index], r_col_expansion.ptr(), 
		    exact_vector_e_column.ptr());

    } // end of iterating over each reference point.
    
    double relative_error = 
      MatrixUtil::EntrywiseNormDifferenceRelative
      (exact_vector_e_column, approx_column, 1);

    max_relative_error = std::max(max_relative_error, relative_error);

  } // end of iterating over each query point.

  printf("Maximum relative error: %g\n", max_relative_error);
}
