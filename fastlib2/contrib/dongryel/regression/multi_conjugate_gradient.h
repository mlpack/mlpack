#ifndef MULTI_CONJUGATE_GRADIENT_H
#define MULTI_CONJUGATE_GRADIENT_H

#include "fastlib/fastlib.h"
#include "fastlib/sparse/trilinos/include/az_blas_wrappers.h"

template<typename TAlgorithm>
class MultiConjugateGradient {
  
private:

  /** @brief The pointer to the query tree.
   */
  typename TAlgorithm::QueryTree *qroot_;

  /** @brief The column-oriented query dataset.
   */
  Matrix qset_;

  /** @brief The reference-dependent weights used for the computation.
   */
  Vector rset_inv_norm_consts_;

  /** @brief The dimension of the matrix to be inverted.
   */
  int row_length_;

  /** @brief The pointer to the algorithm that contains the linear operator.
   */
  TAlgorithm *algorithm_;

  bool BreakDown_(const double *p_vec, const double *linear_transformed_p_vec,
		  double p_vec_dot_linear_transformed_p_vec) {
    double p_vec_norm = la::LengthEuclidean(row_length_, p_vec);
    double linear_transformed_p_vec_norm = 
      la::LengthEuclidean(row_length_, linear_transformed_p_vec);
    
    return (fabs(p_vec_dot_linear_transformed_p_vec) <=
	    100.0 * p_vec_norm * linear_transformed_p_vec_norm * DBL_EPSILON);
  }

  void ComputeResiduals_(const ArrayList<bool> &query_in_cg_loop,
			 const Matrix &right_hand_sides, 
			 const Matrix &current_solutions, Matrix &residuals) {

    // Multiply the current solutions by the linear operator.
    algorithm_->LinearOperator(qroot_, qset_, query_in_cg_loop,
			       current_solutions, residuals);

    // Compute the residuals by subtracting from b in Ax = b.
    for(index_t i = 0; i < residuals.n_cols(); i++) {
      for(index_t j = 0; j < residuals.n_rows(); j++) {
	residuals.set(j, i, right_hand_sides.get(j, i) - residuals.get(j, i));
      }
    }
  }

  /** @brief Computes the residual norm, the residual norm divided by
   *         the right hand side norm, and the dot product between the
   *         z vector and residual vector for a single query.
   */
  void ComputeGlobalScalar_
  (int q, const Matrix &right_hand_sides, const Matrix &z_vecs, 
   const Matrix &residuals, Vector &residual_norms, 
   Vector &scaled_residual_norms, Vector &r_z_dots) {

    residual_norms[q] = la::LengthEuclidean(row_length_,
					    residuals.GetColumnPtr(q));
    scaled_residual_norms[q] = residual_norms[q] /
      la::LengthEuclidean(row_length_, right_hand_sides.GetColumnPtr(q));
    r_z_dots[q] = la::Dot(row_length_, residuals.GetColumnPtr(q),
			  z_vecs.GetColumnPtr(q));
  }

  /** @brief Computes the residual norm, the residual norm divided by
   *         the right hand side norm, and the dot product between the
   *         z vectors and residual vectors.
   */
  void ComputeGlobalScalars_
  (const Matrix &right_hand_sides, const Matrix &z_vecs, 
   const Matrix &residuals, Vector &residual_norms, 
   Vector &scaled_residual_norms, Vector &r_z_dots) {

    for(index_t q = 0; q < right_hand_sides.n_cols(); q++) {
      residual_norms[q] = la::LengthEuclidean(row_length_,
					      residuals.GetColumnPtr(q));
      scaled_residual_norms[q] = residual_norms[q] /
	la::LengthEuclidean(row_length_, right_hand_sides.GetColumnPtr(q));
      r_z_dots[q] = la::Dot(row_length_, residuals.GetColumnPtr(q),
			    z_vecs.GetColumnPtr(q));
    }
  }

 public:
  
  MultiConjugateGradient() {}

  ~MultiConjugateGradient() {}

  void Init(typename TAlgorithm::QueryTree *qroot_in, const Matrix &qset_in,
	    const Vector &rset_inv_norm_consts_in, int row_length_in,
	    TAlgorithm *algorithm_in) {

    qroot_ = qroot_in;
    qset_.Alias(qset_in);
    rset_inv_norm_consts_.Alias(rset_inv_norm_consts_in);
    row_length_ = row_length_in;
    algorithm_ = algorithm_in;
  }

  void Iterate(const Matrix &right_hand_sides, Matrix &solutions) {

    // This is for storing the flag for each query point - whether it
    // is in the outer CG loop or not.
    ArrayList<bool> query_in_cg_loop;
    query_in_cg_loop.Init(solutions.n_cols());
    int num_queries_in_cg_loop = solutions.n_cols();
    for(index_t q = 0; q < solutions.n_cols(); q++) {
      query_in_cg_loop[q] = true;
    }
    
    // Stores the break-down tolerance for each query point.
    Vector brkdown_tol;
    brkdown_tol.Init(solutions.n_cols());
    brkdown_tol.SetAll(DBL_EPSILON);

    double alpha;
    double p_ap_dot;
    
    // Temporary space to store the residuals
    Matrix residuals;
    Vector true_scaled_residuals;
    residuals.Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
    true_scaled_residuals.Init(right_hand_sides.n_cols());

    // More temporary variables for storing intermediate computation
    // results.
    Matrix p_vecs;
    p_vecs.Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
    Matrix z_vecs;
    z_vecs.Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
    Matrix linear_transformed_p_vecs;
    linear_transformed_p_vecs.Init(right_hand_sides.n_rows(),
				   right_hand_sides.n_cols());
    Vector residual_norms, scaled_residual_norms, r_z_dots, r_z_dot_olds;
    residual_norms.Init(right_hand_sides.n_cols());
    scaled_residual_norms.Init(right_hand_sides.n_cols());
    r_z_dots.Init(right_hand_sides.n_cols());
    r_z_dot_olds.Init(right_hand_sides.n_cols());

    // p = 0
    p_vecs.SetZero();

    // beta = 0
    Vector beta_vec;
    beta_vec.Init(right_hand_sides.n_cols());
    beta_vec.SetZero();

    // Compute the initial residual based on the initial guess. Since
    //we assume that the initial guesses are all zero vectors, we can
    //just assume that the initial residuals are the right hand sides.
    residuals.CopyValues(right_hand_sides);

    //  z = M r - this is assuming no preconditioner.
    z_vecs.CopyValues(residuals);
    
    // Compute a few global scalars:
    //     1) ||r||
    //     2) scaled ||r|| (i.e. ||r|| / ||b||)
    //     3) r_z_dot = <z, r>
    ComputeGlobalScalars_(right_hand_sides, z_vecs, residuals,
			  residual_norms, scaled_residual_norms, r_z_dots);
    
    // Make a copy of the true residual norms.
    true_scaled_residuals.CopyValues(scaled_residual_norms);

    // flag that denotes whether the linear operator has been called before.
    bool called_for_first_time = true;

    // Start the main loop of the CG iteration.
    for(index_t iter = 1; iter <= row_length_ && num_queries_in_cg_loop > 0; 
	iter++) {
      
      printf("%d queries are in the CG loop...\n", num_queries_in_cg_loop);

      // p  = z + beta * p
      for(index_t q = 0; q < p_vecs.n_cols(); q++) {
	// p = beta * p
	la::Scale(row_length_, beta_vec[q], p_vecs.GetColumnPtr(q));
	la::AddTo(row_length_, z_vecs.GetColumnPtr(q), p_vecs.GetColumnPtr(q));
      }

      // ap = A p: applies the linear operator to each query point
      // simultaneously.
      algorithm_->LinearOperator
	(qroot_, qset_, query_in_cg_loop, p_vecs, linear_transformed_p_vecs,
	 called_for_first_time);
      
      // Now loop over each query point.
      for(index_t q = 0; q < solutions.n_cols(); q++) {

	// If the current query has finished, skip it.
	if(!query_in_cg_loop[q]) {
	  continue;
	}

	// Compute p^T A p for each query.
	p_ap_dot = la::Dot(row_length_, p_vecs.GetColumnPtr(q),
			   linear_transformed_p_vecs.GetColumnPtr(q));

	// If we are losing symmetric positive definiteness, then we
	// should stop updating solution for this query point.
	if (p_ap_dot < brkdown_tol[q]) {
	  
	  if(p_ap_dot < 0 || 
	     BreakDown_(p_vecs.GetColumnPtr(q), 
			linear_transformed_p_vecs.GetColumnPtr(q), p_ap_dot)) {
	    query_in_cg_loop[q] = false;
	    num_queries_in_cg_loop--;
	  }

	  // Otherwise, readjust breakdown tolerance according to the
	  // current postive-definiteness of the matrix.
	  else {
	    brkdown_tol[q] = 0.1 * p_ap_dot;
	  }
	}
	
	alpha  = r_z_dots[q] / p_ap_dot;

	// x = x + alpha * p
	la::AddExpert(row_length_, alpha, p_vecs.GetColumnPtr(q),
		      solutions.GetColumnPtr(q));

	// r = r - alpha * Ap
	la::AddExpert(row_length_, -alpha,
		      linear_transformed_p_vecs.GetColumnPtr(q),
		      residuals.GetColumnPtr(q));

	// z = M^-1 r - currently no preconditioner is used.
	la::ScaleOverwrite(row_length_, 1, residuals.GetColumnPtr(q),
			   z_vecs.GetColumnPtr(q));
	
	r_z_dot_olds[q] = r_z_dots[q];

	// Compute a few global scalars:
	//     1) ||r||
	//     2) scaled ||r|| (i.e. ||r|| / ||b||)
	//     3) r_z_dot = <z, r>
	ComputeGlobalScalar_(q, right_hand_sides, z_vecs, residuals,
			     residual_norms, scaled_residual_norms, r_z_dots);
	
	if(!query_in_cg_loop[q]) {
	  break;
	}
	
	beta_vec[q] = r_z_dots[q] / r_z_dot_olds[q];
	
	// This points to another possible problem...
	if (fabs(r_z_dots[q]) < brkdown_tol[q]) {
	  if(BreakDown_(residuals.GetColumnPtr(q), z_vecs.GetColumnPtr(q),
			r_z_dots[q])) {
	    query_in_cg_loop[q] = false;
	    num_queries_in_cg_loop--;
	  }
	  else {
	    brkdown_tol[q] = 0.1 * fabs(r_z_dots[q]);
	  }
	}

	// Now check whether the current query point has converged...
	if(scaled_residual_norms[q] < 1e-2) {
	  query_in_cg_loop[q] = false;
	  num_queries_in_cg_loop--;
	}

      } // end of iterating over each query point.

    } // end of iterating over each iteration of CG.
  }
  
};

#endif
