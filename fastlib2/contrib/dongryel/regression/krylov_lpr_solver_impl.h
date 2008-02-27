// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

#include "multi_conjugate_gradient.h"

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::LinearOperator
(QueryTree *qroot, const Matrix &qset, 
 const ArrayList<bool> &query_in_cg_loop, const Matrix &original_vectors, 
 Matrix &linear_transformed_vectors) {

  Matrix vector_l, vector_e;
  Vector vector_used_error, vector_n_pruned;
  
  vector_l.Init(row_length_, original_vectors.n_cols());
  vector_e.Init(row_length_, original_vectors.n_cols());
  vector_used_error.Init(original_vectors.n_cols());
  vector_n_pruned.Init(original_vectors.n_cols());
  
  // Initialize the multivector to zero.
  linear_transformed_vectors.SetZero();
  
  for(index_t d = 0; d < row_length_; d++) {
    ComputeWeightedVectorSum_
      (qroot, qset, rset_inv_norm_consts_, &query_in_cg_loop, d,
       vector_l, vector_e, vector_used_error, vector_n_pruned);
    
    // Accumulate the product between the computed vector and each
    // scalar component of the X.
    for(index_t q = 0; q < qset.n_cols(); q++) {

      // If the current query is not in the CG loop, we don't have to
      // update the vector components for it!
      if(!query_in_cg_loop[q]) {
	continue;
      }

      for(index_t j = 0; j < row_length_; j++) {
	linear_transformed_vectors.set
	  (j, q, linear_transformed_vectors.get(j, q) +
	   original_vectors.get(d, q) * vector_e.get(j, q));
      }
    } // end of iterating over each query.
  } // end of iterating over each component.
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::SolveLinearProblems_
(QueryTree *qroot, const Matrix &qset, const Matrix &right_hand_sides_e,
 Matrix &solution_vectors_e) {
  
  MultiConjugateGradient<KrylovLpr<TKernel, TPruneRule> > mcg_algorithm;
  mcg_algorithm.Init(qroot, qset, rset_inv_norm_consts_, row_length_, this);

  // Initialize the solution vectors to be zero.
  solution_vectors_e.SetZero();
  mcg_algorithm.Iterate(right_hand_sides_e, solution_vectors_e);
}
