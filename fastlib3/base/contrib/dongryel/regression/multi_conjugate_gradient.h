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
   Vector &scaled_residual_norms, Vector &r_z_dots,
   Matrix *loo_right_hand_sides, Matrix *loo_z_vecs, 
   Matrix *loo_residuals, Vector *loo_residual_norms, 
   Vector *loo_scaled_residual_norms, Vector *loo_r_z_dots,
   const Matrix &expansion_right_hand_sides, const Matrix &expansion_z_vecs, 
   const Matrix &expansion_residuals, Vector &expansion_residual_norms, 
   Vector &expansion_scaled_residual_norms, Vector &expansion_r_z_dots) {

    residual_norms[q] = la::LengthEuclidean(row_length_,
					    residuals.GetColumnPtr(q));
    scaled_residual_norms[q] = residual_norms[q] /
      la::LengthEuclidean(row_length_, right_hand_sides.GetColumnPtr(q));
    r_z_dots[q] = la::Dot(row_length_, residuals.GetColumnPtr(q),
			  z_vecs.GetColumnPtr(q));

    if(loo_residual_norms != NULL) {
      (*loo_residual_norms)[q] = la::LengthEuclidean
	(row_length_, loo_residuals->GetColumnPtr(q));
      (*loo_scaled_residual_norms)[q] = (*loo_residual_norms)[q] /
	la::LengthEuclidean(row_length_, 
			    loo_right_hand_sides->GetColumnPtr(q));
      (*loo_r_z_dots)[q] = la::Dot(row_length_, loo_residuals->GetColumnPtr(q),
				   loo_z_vecs->GetColumnPtr(q));
    }

    expansion_residual_norms[q] = la::LengthEuclidean
      (row_length_, expansion_residuals.GetColumnPtr(q));
    expansion_scaled_residual_norms[q] = expansion_residual_norms[q] /
      la::LengthEuclidean(row_length_, expansion_right_hand_sides.
			  GetColumnPtr(q));
    expansion_r_z_dots[q] = la::Dot(row_length_, 
				    expansion_residuals.GetColumnPtr(q),
				    expansion_z_vecs.GetColumnPtr(q));
  }

  /** @brief Computes the residual norm, the residual norm divided by
   *         the right hand side norm, and the dot product between the
   *         z vectors and residual vectors.
   */
  void ComputeGlobalScalars_
  (const Matrix &right_hand_sides, const Matrix &z_vecs, 
   const Matrix &residuals, Vector &residual_norms, 
   Vector &scaled_residual_norms, Vector &r_z_dots,
   Matrix *loo_right_hand_sides, Matrix *loo_z_vecs, 
   Matrix *loo_residuals, Vector *loo_residual_norms, 
   Vector *loo_scaled_residual_norms, Vector *loo_r_z_dots,
   const Matrix &expansion_right_hand_sides, const Matrix &expansion_z_vecs, 
   const Matrix &expansion_residuals, Vector &expansion_residual_norms, 
   Vector &expansion_scaled_residual_norms, Vector &expansion_r_z_dots) {

    for(index_t q = 0; q < right_hand_sides.n_cols(); q++) {
      residual_norms[q] = la::LengthEuclidean(row_length_,
					      residuals.GetColumnPtr(q));
      scaled_residual_norms[q] = residual_norms[q] /
	la::LengthEuclidean(row_length_, right_hand_sides.GetColumnPtr(q));
      r_z_dots[q] = la::Dot(row_length_, residuals.GetColumnPtr(q),
			    z_vecs.GetColumnPtr(q));

      if(loo_residual_norms != NULL) {
	(*loo_residual_norms)[q] = la::LengthEuclidean
	  (row_length_, loo_residuals->GetColumnPtr(q));
	(*loo_scaled_residual_norms)[q] = (*loo_residual_norms)[q] /
	  la::LengthEuclidean(row_length_, 
			      loo_right_hand_sides->GetColumnPtr(q));
	(*loo_r_z_dots)[q] = la::Dot(row_length_, 
				     loo_residuals->GetColumnPtr(q),
				     loo_z_vecs->GetColumnPtr(q));
      }
      
      expansion_residual_norms[q] = la::LengthEuclidean
	(row_length_, expansion_residuals.GetColumnPtr(q));
      expansion_scaled_residual_norms[q] = expansion_residual_norms[q] /
	la::LengthEuclidean(row_length_, expansion_right_hand_sides.
			    GetColumnPtr(q));
      expansion_r_z_dots[q] = la::Dot(row_length_, 
				      expansion_residuals.GetColumnPtr(q),
				      expansion_z_vecs.GetColumnPtr(q));
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

  void Iterate(const Matrix &right_hand_sides, Matrix *loo_right_hand_sides, 
	       const Matrix &query_expansions, Matrix &solutions,
	       Matrix *loo_solutions, Matrix &query_expansion_solutions) {

    // This is for storing the flag for each query point - whether it
    // is in the outer CG loop or not.
    ArrayList<bool> query_in_cg_loop, *loo_query_in_cg_loop = NULL,
      expansion_query_in_cg_loop;
    query_in_cg_loop.Init(solutions.n_cols());
    expansion_query_in_cg_loop.Init(solutions.n_cols());
    int num_queries_in_cg_loop = solutions.n_cols();
    for(index_t q = 0; q < solutions.n_cols(); q++) {
      query_in_cg_loop[q] = expansion_query_in_cg_loop[q] = true;
    }
    
    // Stores the break-down tolerance for each query point.
    Vector brkdown_tol, *loo_brkdown_tol = NULL,
      expansion_brkdown_tol;
    brkdown_tol.Init(solutions.n_cols());
    expansion_brkdown_tol.Init(solutions.n_cols());
    brkdown_tol.SetAll(DBL_EPSILON);
    expansion_brkdown_tol.SetAll(DBL_EPSILON);

    double alpha = 0, loo_alpha = 0, expansion_alpha = 0;
    double p_ap_dot = 0, loo_p_ap_dot = 0, expansion_p_ap_dot = 0;
    
    // Temporary space to store the residuals
    Matrix residuals, *loo_residuals = NULL, expansion_residuals;
    residuals.Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
    expansion_residuals.Init(right_hand_sides.n_rows(), 
			     right_hand_sides.n_cols());

    // More temporary variables for storing intermediate computation
    // results.
    Matrix p_vecs, *loo_p_vecs = NULL, expansion_p_vecs;
    p_vecs.Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
    expansion_p_vecs.Init(right_hand_sides.n_rows(), 
			  right_hand_sides.n_cols());
    Matrix z_vecs, *loo_z_vecs = NULL, expansion_z_vecs;
    z_vecs.Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
    expansion_z_vecs.Init(right_hand_sides.n_rows(),
			  right_hand_sides.n_cols());
    Matrix linear_transformed_p_vecs, *linear_transformed_loo_p_vecs = NULL,
      linear_transformed_expansion_p_vecs;
    linear_transformed_p_vecs.Init(right_hand_sides.n_rows(),
				   right_hand_sides.n_cols());
    linear_transformed_expansion_p_vecs.Init(right_hand_sides.n_rows(),
					     right_hand_sides.n_cols());
    Vector beta_vec, *loo_beta_vec = NULL, expansion_beta_vec;
    beta_vec.Init(right_hand_sides.n_cols());
    expansion_beta_vec.Init(right_hand_sides.n_cols());

    Vector residual_norms, scaled_residual_norms, r_z_dots, r_z_dot_olds,
      *loo_residual_norms = NULL, *loo_scaled_residual_norms = NULL,
      *loo_r_z_dots = NULL, *loo_r_z_dot_olds = NULL,
      expansion_residual_norms, expansion_scaled_residual_norms,
      expansion_r_z_dots, expansion_r_z_dot_olds;
    residual_norms.Init(right_hand_sides.n_cols());
    scaled_residual_norms.Init(right_hand_sides.n_cols());
    r_z_dots.Init(right_hand_sides.n_cols());
    r_z_dot_olds.Init(right_hand_sides.n_cols());
    expansion_residual_norms.Init(right_hand_sides.n_cols());
    expansion_scaled_residual_norms.Init(right_hand_sides.n_cols());
    expansion_r_z_dots.Init(right_hand_sides.n_cols());
    expansion_r_z_dot_olds.Init(right_hand_sides.n_cols());    

    // Allocate stuffs for leave-one-out estimates.
    if(loo_right_hand_sides != NULL) {
      loo_query_in_cg_loop = new ArrayList<bool>();
      loo_query_in_cg_loop->Init(right_hand_sides.n_cols());
      for(index_t i = 0; i < right_hand_sides.n_cols(); i++) {
	(*loo_query_in_cg_loop)[i] = true;
      }
      loo_brkdown_tol = new Vector();
      loo_brkdown_tol->Init(right_hand_sides.n_cols());
      loo_brkdown_tol->SetAll(DBL_EPSILON);
      loo_residuals = new Matrix();
      loo_residuals->Init(right_hand_sides.n_rows(), 
			  right_hand_sides.n_cols());
      loo_p_vecs = new Matrix();
      loo_p_vecs->Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
      loo_z_vecs = new Matrix();
      loo_z_vecs->Init(right_hand_sides.n_rows(), right_hand_sides.n_cols());
      linear_transformed_loo_p_vecs = new Matrix();
      linear_transformed_loo_p_vecs->Init(right_hand_sides.n_rows(), 
					  right_hand_sides.n_cols());
      loo_beta_vec = new Vector();
      loo_beta_vec->Init(right_hand_sides.n_cols());
      loo_residual_norms = new Vector();
      loo_residual_norms->Init(right_hand_sides.n_cols());
      loo_scaled_residual_norms = new Vector();
      loo_scaled_residual_norms->Init(right_hand_sides.n_cols());
      loo_r_z_dots = new Vector();
      loo_r_z_dots->Init(right_hand_sides.n_cols());
      loo_r_z_dot_olds = new Vector();
      loo_r_z_dot_olds->Init(right_hand_sides.n_cols());
    }

    // p = 0
    p_vecs.SetZero();
    expansion_p_vecs.SetZero();
    if(loo_p_vecs != NULL) {
      loo_p_vecs->SetZero();
    }

    // beta = 0
    beta_vec.SetZero();
    expansion_beta_vec.SetZero();
    if(loo_beta_vec != NULL) {
      loo_beta_vec->SetZero();
    }

    // Compute the initial residual based on the initial guess. Since
    // we assume that the initial guesses are all zero vectors, we can
    // just assume that the initial residuals are the right hand
    // sides.
    residuals.CopyValues(right_hand_sides);
    expansion_residuals.CopyValues(query_expansions);
    if(loo_residuals != NULL) {
      loo_residuals->CopyValues(*loo_right_hand_sides);
    }

    //  z = M r - this is assuming no preconditioner.
    z_vecs.CopyValues(residuals);
    expansion_z_vecs.CopyValues(expansion_residuals);
    if(loo_z_vecs != NULL) {
      loo_z_vecs->CopyValues(*loo_residuals);
    }
    
    // Compute a few global scalars:
    //     1) ||r||
    //     2) scaled ||r|| (i.e. ||r|| / ||b||)
    //     3) r_z_dot = <z, r>
    ComputeGlobalScalars_(right_hand_sides, z_vecs, residuals,
			  residual_norms, scaled_residual_norms, r_z_dots,
			  loo_right_hand_sides, loo_z_vecs, loo_residuals,
			  loo_residual_norms, loo_scaled_residual_norms, 
			  loo_r_z_dots, query_expansions, expansion_z_vecs, 
			  expansion_residuals, expansion_residual_norms, 
			  expansion_scaled_residual_norms, expansion_r_z_dots);

    // Start the main loop of the CG iteration.
    for(index_t iter = 1; iter <= row_length_ && num_queries_in_cg_loop > 0; 
	iter++) {
      
      printf("%d queries are in the CG loop...\n", num_queries_in_cg_loop);

      // p  = z + beta * p
      for(index_t q = 0; q < p_vecs.n_cols(); q++) {
	// p = beta * p
	la::Scale(row_length_, beta_vec[q], p_vecs.GetColumnPtr(q));
	la::Scale(row_length_, expansion_beta_vec[q],
		  expansion_p_vecs.GetColumnPtr(q));
	// p += z;
	la::AddTo(row_length_, z_vecs.GetColumnPtr(q), p_vecs.GetColumnPtr(q));
	la::AddTo(row_length_, expansion_z_vecs.GetColumnPtr(q),
		  expansion_p_vecs.GetColumnPtr(q));

	if(loo_beta_vec != NULL) {
	  la::Scale(row_length_, (*loo_beta_vec)[q], 
		    loo_p_vecs->GetColumnPtr(q));
	  la::AddTo(row_length_, loo_z_vecs->GetColumnPtr(q), 
		    loo_p_vecs->GetColumnPtr(q));
	}
      }

      // ap = A p: applies the linear operator to each query point
      // simultaneously.
      algorithm_->LinearOperator
	(qroot_, qset_, query_in_cg_loop, p_vecs, loo_p_vecs, expansion_p_vecs,
	 linear_transformed_p_vecs, linear_transformed_loo_p_vecs,
	 linear_transformed_expansion_p_vecs);
      
      // Now loop over each query point.
      for(index_t q = 0; q < solutions.n_cols(); q++) {

	// If the current query has finished, skip it.
	if(!query_in_cg_loop[q]) {
	  continue;
	}

	// Compute p^T A p for each query.
	p_ap_dot = la::Dot(row_length_, p_vecs.GetColumnPtr(q),
			   linear_transformed_p_vecs.GetColumnPtr(q));
	expansion_p_ap_dot = 
	  la::Dot(row_length_, expansion_p_vecs.GetColumnPtr(q),
		  linear_transformed_expansion_p_vecs.GetColumnPtr(q));
	if(loo_p_vecs != NULL) {
	  loo_p_ap_dot = la::Dot(row_length_, loo_p_vecs->GetColumnPtr(q),
				 linear_transformed_loo_p_vecs->
				 GetColumnPtr(q));
	}

	// If we are losing symmetric positive definiteness, then we
	// should stop updating solution for this query point.
	if(query_in_cg_loop[q] && p_ap_dot < brkdown_tol[q]) {
	  
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
	if(expansion_query_in_cg_loop[q] &&
	   expansion_p_ap_dot < expansion_brkdown_tol[q]) {
	  
	  if(expansion_p_ap_dot < 0 || 
	     BreakDown_(expansion_p_vecs.GetColumnPtr(q), 
			linear_transformed_expansion_p_vecs.GetColumnPtr(q), 
			expansion_p_ap_dot)) {
	    expansion_query_in_cg_loop[q] = false;
	  }

	  // Otherwise, readjust breakdown tolerance according to the
	  // current postive-definiteness of the matrix.
	  else {
	    expansion_brkdown_tol[q] = 0.1 * expansion_p_ap_dot;
	  }
	}
	if(loo_solutions != NULL && (*loo_query_in_cg_loop)[q]) {
	  
	  // If we are losing symmetric positive definiteness for the
	  // leave-one-out estimates, then we should stop updating
	  // solution for this query point.
	  if (loo_p_ap_dot < (*loo_brkdown_tol)[q]) {
	    
	    if(loo_p_ap_dot < 0 || 
	       BreakDown_(loo_p_vecs->GetColumnPtr(q), 
			  linear_transformed_loo_p_vecs->GetColumnPtr(q), 
			  loo_p_ap_dot)) {
	      (*loo_query_in_cg_loop)[q] = false;
	    }
	    
	    // Otherwise, readjust breakdown tolerance according to the
	    // current postive-definiteness of the matrix.
	    else {
	      (*loo_brkdown_tol)[q] = 0.1 * loo_p_ap_dot;
	    }
	  }
	}
	
	// Compute alpha
	alpha  = r_z_dots[q] / p_ap_dot;
	expansion_alpha = expansion_r_z_dots[q] / expansion_p_ap_dot;
	if(loo_r_z_dots != NULL) {
	  loo_alpha = (*loo_r_z_dots)[q] / loo_p_ap_dot;
	}

	// x = x + alpha * p
	if(query_in_cg_loop[q]) {
	  la::AddExpert(row_length_, alpha, p_vecs.GetColumnPtr(q),
			solutions.GetColumnPtr(q));
	}
	if(expansion_query_in_cg_loop[q]) {
	  la::AddExpert(row_length_, expansion_alpha, 
			expansion_p_vecs.GetColumnPtr(q),
			query_expansion_solutions.GetColumnPtr(q));
	}
	if(loo_solutions != NULL && (*loo_query_in_cg_loop)[q]) {
	  la::AddExpert(row_length_, loo_alpha, loo_p_vecs->GetColumnPtr(q),
			loo_solutions->GetColumnPtr(q));
	}

	// r = r - alpha * Ap
	la::AddExpert(row_length_, -alpha,
		      linear_transformed_p_vecs.GetColumnPtr(q),
		      residuals.GetColumnPtr(q));
	la::AddExpert(row_length_, -expansion_alpha,
		      linear_transformed_expansion_p_vecs.GetColumnPtr(q),
		      expansion_residuals.GetColumnPtr(q));
	if(loo_residuals != NULL) {
	  la::AddExpert(row_length_, -loo_alpha,
			linear_transformed_loo_p_vecs->GetColumnPtr(q),
			loo_residuals->GetColumnPtr(q));
	}

	// z = M^-1 r - currently no preconditioner is used.
	la::ScaleOverwrite(row_length_, 1, residuals.GetColumnPtr(q),
			   z_vecs.GetColumnPtr(q));
	la::ScaleOverwrite(row_length_, 1, expansion_residuals.GetColumnPtr(q),
			   expansion_z_vecs.GetColumnPtr(q));
	if(loo_residuals != NULL) {
	  la::ScaleOverwrite(row_length_, 1, loo_residuals->GetColumnPtr(q),
			     loo_z_vecs->GetColumnPtr(q));
	}

	// Save the old values of dot product between the residual and
	// z vector.
	r_z_dot_olds[q] = r_z_dots[q];
	expansion_r_z_dot_olds[q] = expansion_r_z_dots[q];
	if(loo_r_z_dot_olds != NULL) {
	  (*loo_r_z_dot_olds)[q] = (*loo_r_z_dots)[q];
	}

	// Compute a few global scalars:
	//     1) ||r||
	//     2) scaled ||r|| (i.e. ||r|| / ||b||)
	//     3) r_z_dot = <z, r>
	ComputeGlobalScalar_(q, right_hand_sides, z_vecs, residuals,
			     residual_norms, scaled_residual_norms, r_z_dots,
			     loo_right_hand_sides, loo_z_vecs, loo_residuals,
			     loo_residual_norms, loo_scaled_residual_norms, 
			     loo_r_z_dots, query_expansions, 
			     expansion_z_vecs, expansion_residuals,
			     expansion_residual_norms, 
			     expansion_scaled_residual_norms, 
			     expansion_r_z_dots);
	
	if(!query_in_cg_loop[q]) {
	  continue;
	}
	
	// Compute beta
	beta_vec[q] = r_z_dots[q] / r_z_dot_olds[q];
	expansion_beta_vec[q] = expansion_r_z_dots[q] / 
	  expansion_r_z_dot_olds[q];
	if(loo_beta_vec != NULL) {
	  (*loo_beta_vec)[q] = (*loo_r_z_dots)[q] / (*loo_r_z_dot_olds)[q];
	}
	
	// This points to another possible problem...
	if (query_in_cg_loop[q] && fabs(r_z_dots[q]) < brkdown_tol[q]) {
	  if(BreakDown_(residuals.GetColumnPtr(q), z_vecs.GetColumnPtr(q),
			r_z_dots[q])) {
	    query_in_cg_loop[q] = false;
	    num_queries_in_cg_loop--;
	  }
	  else {
	    brkdown_tol[q] = 0.1 * fabs(r_z_dots[q]);
	  }
	}
	if(expansion_query_in_cg_loop[q] &&
	   fabs(expansion_r_z_dots[q]) < expansion_brkdown_tol[q]) {
	  if(BreakDown_(expansion_residuals.GetColumnPtr(q), 
			expansion_z_vecs.GetColumnPtr(q),
			expansion_r_z_dots[q])) {
	    expansion_query_in_cg_loop[q] = false;
	  }
	  else {
	    expansion_brkdown_tol[q] = 0.1 * fabs(expansion_r_z_dots[q]);
	  }
	}
	if(loo_r_z_dots != NULL &&
	   (*loo_query_in_cg_loop)[q]) {
	  if (fabs((*loo_r_z_dots)[q]) < (*loo_brkdown_tol)[q]) {
	    if(BreakDown_(loo_residuals->GetColumnPtr(q), 
			  loo_z_vecs->GetColumnPtr(q),
			  (*loo_r_z_dots)[q])) {
	      (*loo_query_in_cg_loop)[q] = false;
	    }
	    else {
	      (*loo_brkdown_tol)[q] = 0.1 * fabs((*loo_r_z_dots)[q]);
	    }
	  }
	}
	
	// Now check whether the current query point has converged...
	if(query_in_cg_loop[q] && scaled_residual_norms[q] < 1e-3) {
	  query_in_cg_loop[q] = false;
	  num_queries_in_cg_loop--;
	}

      } // end of iterating over each query point.

    } // end of iterating over each iteration of CG.

    // Memory cleanup
    if(loo_right_hand_sides != NULL) {
      delete loo_query_in_cg_loop;
      delete loo_brkdown_tol;
      delete loo_residuals;
      delete loo_p_vecs;
      delete loo_z_vecs;
      delete linear_transformed_loo_p_vecs;
      delete loo_beta_vec;
      delete loo_residual_norms;
      delete loo_scaled_residual_norms;
      delete loo_r_z_dots;
      delete loo_r_z_dot_olds;
    }
  }
  
};

#endif
