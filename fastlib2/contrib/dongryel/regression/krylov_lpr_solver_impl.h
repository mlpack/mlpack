// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_KRYLOV_LPR_H
#error "This file is not a public header file!"
#endif

#include "multi_conjugate_gradient.h"

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::LinearOperatorConfidenceBand
(QueryTree *qroot, const Matrix &qset, 
 const Matrix &query_expansion_solution_vectors,
 Matrix &linear_transformed_query_expansion_solution_vectors) {
  
  Matrix vector_l, vector_e;
  Vector vector_used_error, vector_n_pruned;
  
  vector_l.Init(row_length_, qset.n_cols());
  vector_e.Init(row_length_, qset.n_cols());
  vector_used_error.Init(qset.n_cols());
  vector_n_pruned.Init(qset.n_cols());
  
  // Initialize the multivector to zero.
  linear_transformed_query_expansion_solution_vectors.SetZero();

  for(index_t d = 0; d < row_length_; d++) {

    // Compute the current column linear operator.
    ComputeWeightedVectorSum_
      (qroot, qset, rset_inv_squared_norm_consts_, NULL, true, d,
       vector_l, vector_e, vector_used_error, vector_n_pruned, NULL);

    // Accumulate the product between the computed vector and each
    // scalar component of the X.
    for(index_t q = 0; q < qset.n_cols(); q++) {

      for(index_t j = 0; j < row_length_; j++) {
	linear_transformed_query_expansion_solution_vectors.set
	  (j, q, 
	   linear_transformed_query_expansion_solution_vectors.get(j, q) +
	   query_expansion_solution_vectors.get(d, q) * vector_e.get(j, q));
      }
    } // end of iterating over each query.
  } // end of iterating over each component.  
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::LinearOperator
(QueryTree *qroot, const Matrix &qset, const ArrayList<bool> &query_in_cg_loop,
 const Matrix &original_vectors, Matrix *leave_one_out_original_vectors,
 const Matrix &expansion_original_vectors, Matrix &linear_transformed_vectors, 
 Matrix *linear_transformed_leave_one_out_vectors,
 Matrix &linear_transformed_expansion_vectors) {

  Matrix vector_l, vector_e, *leave_one_out_vector_e = NULL;
  Vector vector_used_error, vector_n_pruned;
  
  vector_l.Init(row_length_, original_vectors.n_cols());
  vector_e.Init(row_length_, original_vectors.n_cols());

  if(leave_one_out_original_vectors != NULL) {
    leave_one_out_vector_e = new Matrix();
    leave_one_out_vector_e->Init(row_length_, original_vectors.n_cols());
    linear_transformed_leave_one_out_vectors->SetZero();
  }
  vector_used_error.Init(original_vectors.n_cols());
  vector_n_pruned.Init(original_vectors.n_cols());
  
  // Initialize the multivector to zero.
  linear_transformed_vectors.SetZero();
  linear_transformed_expansion_vectors.SetZero();
    
  for(index_t d = 0; d < row_length_; d++) {

    // Compute the current column linear operator.
    ComputeWeightedVectorSum_
      (qroot, qset, rset_inv_norm_consts_, &query_in_cg_loop, false, d,
       vector_l, vector_e, vector_used_error, vector_n_pruned,
       leave_one_out_vector_e);

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
	linear_transformed_expansion_vectors.set
	  (j, q, linear_transformed_expansion_vectors.get(j, q) +
	   expansion_original_vectors.get(d, q) * vector_e.get(j, q));

	if(linear_transformed_leave_one_out_vectors != NULL) {
	  linear_transformed_leave_one_out_vectors->set
	    (j, q, linear_transformed_leave_one_out_vectors->get(j, q) +
	     leave_one_out_original_vectors->get(d, q) *
	     leave_one_out_vector_e->get(j, q));
	}
      }
    } // end of iterating over each query.
  } // end of iterating over each component.

  // Memory cleanup
  delete leave_one_out_vector_e;
}

template<typename TKernel, typename TPruneRule>
void KrylovLpr<TKernel, TPruneRule>::SolveLinearProblems_
(QueryTree *qroot, const Matrix &qset, const Matrix &right_hand_sides_e,
 Matrix *leave_one_out_right_hand_sides_e, const Matrix &query_expansions,
 Matrix &solution_vectors_e, Matrix *leave_one_out_solution_vectors_e,
 Matrix &query_expansion_solutions) {
  
  MultiConjugateGradient<KrylovLpr<TKernel, TPruneRule> > mcg_algorithm;
  mcg_algorithm.Init(qroot, qset, rset_inv_norm_consts_, row_length_, this);

  // Initialize the solution vectors to be zero.
  solution_vectors_e.SetZero();
  if(leave_one_out_solution_vectors_e != NULL) {
    leave_one_out_solution_vectors_e->SetZero();
  }
  query_expansion_solutions.SetZero();

  mcg_algorithm.Iterate(right_hand_sides_e, leave_one_out_right_hand_sides_e,
			query_expansions, solution_vectors_e, 
			leave_one_out_solution_vectors_e,
			query_expansion_solutions);
}
