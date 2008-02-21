// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void KrylovLpr<TKernel>::MaximumRelativeErrorInL1Norm_
(const Matrix &qset, const Matrix &exact_vector_e, const Matrix &approximated,
 const ArrayList<bool> *query_should_exit_the_loop) {
  
  double max_relative_error = 0;
  
  for(index_t q = 0; q < qset.n_cols(); q++) {
    
    if((*query_should_exit_the_loop)[q]) {
      continue;
    }

    // get the column vector containing the approximation.
    const double *approx_column = approximated.GetColumnPtr(q);
    
    // get the column vector accumulating the sum.
    const double *exact_vector_e_column = exact_vector_e.GetColumnPtr(q);
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

template<typename TKernel>
void KrylovLpr<TKernel>::TestRightHandSideComputation_
(const Matrix &qset, const Matrix &approximated) {
  
  Matrix exact_vector_e;
  exact_vector_e.Init(approximated.n_rows(), approximated.n_cols());
  exact_vector_e.SetZero();

  for(index_t q = 0; q < qset.n_cols(); q++) {
    
    // get the column vector corresponding to the current query point.
    const double *q_col = qset.GetColumnPtr(q);

    // get the column vector accumulating the sum.
    double *exact_vector_e_column = exact_vector_e.GetColumnPtr(q);

    for(index_t r = 0; r < rset_.n_cols(); r++) {
    
      // get the column vector corresponding to the current reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // get the column vector containing the appropriate weights.
      const double *r_weights = 
	target_weighted_rset_.GetColumnPtr(r);
      
      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[0].EvalUnnormOnSq(dsqd);

      // for each vector component, update the lower/estimate/upper
      // bound quantities.
      for(index_t d = 0; d <= dimension_; d++) {
	exact_vector_e_column[d] += kernel_value * r_weights[d];
      } // end of iterating over each vector component.

    } // end of iterating over each reference point.
      
  } // end of iterating over each query point.

  MaximumRelativeErrorInL1Norm_(exact_vector_e, approximated,
				NULL);
}

template<typename TKernel>
void KrylovLpr<TKernel>::TestKrylovComputation_
(const Matrix &qset, const Matrix &approximated, 
 const Matrix &current_lanczos_vectors,
 const ArrayList<bool> &query_should_exit_the_loop) {
  
  Matrix exact_vector_e;
  exact_vector_e.Init(approximated.n_rows(), approximated.n_cols());
  exact_vector_e.SetZero();

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
    double *exact_vector_e_column = exact_vector_e.GetColumnPtr(q);

    for(index_t r = 0; r < rset_.n_cols(); r++) {
    
      // get the column vector corresponding to the current reference point.
      const double *r_col = rset_.GetColumnPtr(r);
      
      // compute the pairwise squared distance and kernel value.
      double dsqd = la::DistanceSqEuclidean(dimension_, q_col, r_col);
      double kernel_value = kernels_[0].EvalUnnormOnSq(dsqd);

      // Take the dot product between the query point's Lanczos vector
      // and [1 r^T]^T.
      double dot_product = q_lanczos_vector[0];
      for(index_t d = 1; d <= dimension_; d++) {
	dot_product += r_col[d - 1] * q_lanczos_vector[d];
      }
      double front_factor = dot_product * kernel_value;

      // For each vector component,
      exact_vector_e_column[0] += front_factor;

      for(index_t d = 1; d <= dimension_; d++) {
	exact_vector_e_column[d] += front_factor * r_col[d - 1];

      } // end of iterating over each vector component.      

    } // end of iterating over each reference point.
    
  } // end of iterating over each query point.

  MaximumRelativeErrorInL1Norm_(exact_vector_e, approximated,
				&query_should_exit_the_loop);
}
