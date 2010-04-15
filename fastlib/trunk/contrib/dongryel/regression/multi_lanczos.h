#ifndef MULTI_LANCZOS_H
#define MULTI_LANCZOS_H

#include "fastlib/fastlib.h"
#include <trilinos/az_blas_wrappers.h>

template<typename TAlgorithm>
class MultiLanczos {
  
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
  
  MultiLanczos() {}

  ~MultiLanczos() {}

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

    // These boolean arrays tell whether each query has converged or
    // not (each for three different systems we are solving).
    ArrayList<bool> query_in_cg_loop;
    query_in_cg_loop.Init(solutions.n_cols());
    for(index_t i = 0; i < solutions.n_cols(); i++) {
      query_in_cg_loop[i] = true;
    }
    
    // Total number of queries that have not finished computing its
    // solutions.
    int num_queries_in_lanczos_loop = solutions.n_cols();

    // These matrices store the current and previous iteration's
    // residuals.
    Matrix residuals, *loo_residuals = NULL, expansion_residuals,
      previous_residuals, *previous_loo_residuals = NULL,
      previous_expansion_residuals;

    residuals.Init(solutions.n_rows(), solutions.n_cols());
    residuals.SetZero();
    expansion_residuals.Init(solutions.n_rows(), solutions.n_cols());
    expansion_residuals.SetZero();
    previous_residuals.Init(solutions.n_rows(), solutions.n_cols());
    previous_residuals.SetZero();
    previous_expansion_residuals.Init(solutions.n_rows(), solutions.n_cols());
    previous_expansion_residuals.SetZero();

    if(loo_solutions != NULL) {
      loo_residuals = new Matrix();
      loo_residuals->Init(solutions.n_rows(), solutions.n_cols());
      loo_residuals->SetZero();
      previous_loo_residuals = new Matrix();
      previous_loo_residuals->Init(solutions.n_rows(), solutions.n_cols());
      previous_loo_residuals->SetZero();
    }

    // These matrices store the current and previous iteration's
    // Lanczos basis vectors.
    Matrix q_vecs, *loo_q_vecs = NULL, expansion_q_vecs, previous_q_vecs,
      *previous_loo_q_vecs = NULL, previous_expansion_q_vecs;

    q_vecs.Init(solutions.n_rows(), solutions.n_cols());
    q_vecs.SetZero();
    expansion_q_vecs.Init(solutions.n_rows(), solutions.n_cols());
    expansion_q_vecs.SetZero();
    previous_q_vecs.Init(solutions.n_rows(), solutions.n_cols());
    previous_q_vecs.SetZero();
    previous_expansion_q_vecs.Init(solutions.n_rows(), solutions.n_cols());
    previous_expansion_q_vecs.SetZero();

    if(loo_solutions != NULL) {
      loo_q_vecs = new Matrix();
      loo_q_vecs->Init(solutions.n_rows(), solutions.n_cols());
      loo_q_vecs->SetZero();

      previous_loo_q_vecs = new Matrix();
      previous_loo_q_vecs->Init(solutions.n_rows(), solutions.n_cols());
      previous_loo_q_vecs->SetZero();
    }

    // These matrices store the linear transformed Lanczos basis
    // vectors.
    Matrix linear_transformed_q_vecs, *linear_transformed_loo_q_vecs = NULL,
      linear_transformed_expansion_q_vecs;

    linear_transformed_q_vecs.Init(solutions.n_rows(), solutions.n_cols());
    linear_transformed_q_vecs.SetZero();
    linear_transformed_expansion_q_vecs.Init(solutions.n_rows(),
					     solutions.n_cols());
    linear_transformed_expansion_q_vecs.SetZero();
    
    if(loo_solutions != NULL) {
      linear_transformed_loo_q_vecs = new Matrix();
      linear_transformed_loo_q_vecs->Init(solutions.n_rows(), 
					  solutions.n_cols());
      linear_transformed_loo_q_vecs->SetZero();
    }

    // beta's for each query point and for each linear system.
    Vector beta_vec, *loo_beta_vec = NULL, expansion_beta_vec,
      previous_beta_vec, *previous_loo_beta_vec = NULL, 
      previous_expansion_beta_vec;
    
    beta_vec.Init(solutions.n_cols());
    beta_vec.SetZero();
    expansion_beta_vec.Init(solutions.n_cols());
    expansion_beta_vec.SetZero();
    previous_beta_vec.Init(solutions.n_cols());
    previous_beta_vec.SetZero();
    previous_expansion_beta_vec.Init(solutions.n_cols());
    previous_expansion_beta_vec.SetZero();
    
    if(loo_solutions != NULL) {
      loo_beta_vec = new Vector();
      loo_beta_vec->Init(solutions.n_cols());
      loo_beta_vec->SetZero();
      previous_loo_beta_vec = new Vector();
      previous_loo_beta_vec->Init(solutions.n_cols());
      previous_loo_beta_vec->SetZero();
    }

    // alpha's for each query point and for each linear system.
    Vector alpha_vec, *loo_alpha_vec = NULL, expansion_alpha_vec;
    
    alpha_vec.Init(solutions.n_cols());
    alpha_vec.SetZero();
    expansion_alpha_vec.Init(solutions.n_cols());
    expansion_alpha_vec.SetZero();

    if(loo_solutions != NULL) {
      loo_alpha_vec = new Vector();
      loo_alpha_vec->Init(solutions.n_cols());
      loo_alpha_vec->SetZero();
    }

    // c's for each query point and for each linear system.
    Vector c_vec, *loo_c_vec = NULL, expansion_c_vec,
      previous_c_vec, *previous_loo_c_vec = NULL, previous_expansion_c_vec;
    
    c_vec.Init(solutions.n_cols());
    c_vec.SetZero();
    expansion_c_vec.Init(solutions.n_cols());
    expansion_c_vec.SetZero();
    previous_c_vec.Init(solutions.n_cols());
    previous_c_vec.SetZero();
    previous_expansion_c_vec.Init(solutions.n_cols());
    previous_expansion_c_vec.SetZero();

    if(loo_solutions != NULL) {
      loo_c_vec = new Vector();
      loo_c_vec->Init(solutions.n_cols());
      loo_c_vec->SetZero();
      previous_loo_c_vec = new Vector();
      previous_loo_c_vec->Init(solutions.n_cols());
      previous_loo_c_vec->SetZero();
    }

    // delta's for each query point and for each linear system.
    Vector delta_vec, *loo_delta_vec = NULL, expansion_delta_vec,
      previous_delta_vec, *previous_loo_delta_vec = NULL,
      previous_expansion_delta_vec;
    
    delta_vec.Init(solutions.n_cols());
    delta_vec.SetZero();
    expansion_delta_vec.Init(solutions.n_cols());
    expansion_delta_vec.SetZero();
    previous_delta_vec.Init(solutions.n_cols());
    previous_delta_vec.SetZero();
    previous_expansion_delta_vec.Init(solutions.n_cols());
    previous_expansion_delta_vec.SetZero();

    if(loo_solutions != NULL) {
      loo_delta_vec = new Vector();
      loo_delta_vec->Init(solutions.n_cols());
      loo_delta_vec->SetZero();
      previous_loo_delta_vec = new Vector();
      previous_loo_delta_vec->Init(solutions.n_cols());
      previous_loo_delta_vec->SetZero();
    }

    // q bar's for each query point and for each linear system.
    Matrix q_bar_vecs, *loo_q_bar_vecs = NULL, expansion_q_bar_vecs,
      previous_q_bar_vecs, *previous_loo_q_bar_vecs = NULL,
      previous_expansion_q_bar_vecs;
    
    q_bar_vecs.Init(solutions.n_rows(), solutions.n_cols());
    q_bar_vecs.SetZero();
    expansion_q_bar_vecs.Init(solutions.n_rows(), solutions.n_cols());
    expansion_q_bar_vecs.SetZero();
    previous_q_bar_vecs.Init(solutions.n_rows(), solutions.n_cols());
    previous_q_bar_vecs.SetZero();
    previous_expansion_q_bar_vecs.Init(solutions.n_rows(), solutions.n_cols());
    previous_expansion_q_bar_vecs.SetZero();

    if(loo_solutions != NULL) {
      loo_q_bar_vecs = new Matrix();
      loo_q_bar_vecs->Init(solutions.n_rows(), solutions.n_cols());
      loo_q_bar_vecs->SetZero();
      previous_loo_q_bar_vecs = new Matrix();
      previous_loo_q_bar_vecs->Init(solutions.n_rows(), solutions.n_cols());
      previous_loo_q_bar_vecs->SetZero();
    }


    // Set the initial residuals to be the right hand sides of each
    // system and initialize the beta's (the magnitude of each
    // residual vector).
    previous_residuals.CopyValues(right_hand_sides);
    if(loo_solutions != NULL) {
      previous_loo_residuals->CopyValues(*loo_right_hand_sides);
    }
    previous_expansion_residuals.CopyValues(query_expansions);
    for(index_t q = 0; q < solutions.n_cols(); q++) {
      previous_beta_vec[q] =
	la::LengthEuclidean(row_length_, previous_residuals.GetColumnPtr(q));
      if(loo_solutions != NULL) {
	(*previous_loo_beta_vec)[q] =
	  la::LengthEuclidean(row_length_, 
			      previous_loo_residuals->GetColumnPtr(q));
      }
      previous_expansion_beta_vec[q] =
	la::LengthEuclidean(row_length_,
			    previous_expansion_residuals.GetColumnPtr(q));
    }

    // Start the main loop of the Lanczos iteration.
    for(index_t iter = 1; iter <= row_length_ && 
	  num_queries_in_lanczos_loop > 0; iter++) {
      
      printf("%d queries are in the Lanczos loop...\n", 
	     num_queries_in_lanczos_loop);

      // Compute q_j = r_{j-1} / beta_{j-1}
      for(index_t q = 0; q < q_vecs.n_cols(); q++) {

	// For the linear system (B^T W(q) B)^{-1} B^T W(q) Y.
	la::ScaleOverwrite(row_length_, 1.0 / previous_beta_vec[q], 
			   previous_residuals.GetColumnPtr(q),
			   q_vecs.GetColumnPtr(q));

	// For the linear system (B^T W(q) B)^{-1} B^T W(q) Y (the
	// leave-one-out version).
	if(loo_q_vecs != NULL) {
	  la::ScaleOverwrite(row_length_, 1.0 / (*previous_loo_beta_vec)[q],
			     previous_loo_residuals->GetColumnPtr(q),
			     loo_q_vecs->GetColumnPtr(q));
	}

	// For the linear system (B^T W(q) B)^{-1} t(q).
	la::ScaleOverwrite(row_length_, 1.0 / previous_expansion_beta_vec[q],
			   previous_expansion_residuals.GetColumnPtr(q),
			   expansion_q_vecs.GetColumnPtr(q));
      }

      // A q_j: applies the linear operator to each query point
      // simultaneously.
      algorithm_->LinearOperator
	(qroot_, qset_, query_in_cg_loop, q_vecs, loo_q_vecs, expansion_q_vecs,
	 linear_transformed_q_vecs, linear_transformed_loo_q_vecs,
	 linear_transformed_expansion_q_vecs);
      

      // iterate over each query point.
      for(index_t q = 0; q < q_vecs.n_cols(); q++) {

	// If the current query has converged, then skip it.
	if(!query_in_cg_loop[q]) {
	  continue;
	}

	// Now compute the alpha value: alpha = q_j^T A q_j.
	//
	// For the linear system (B^T W(q) B)^{-1} B^T W(q) Y.	
	alpha_vec[q] = 
	  la::Dot(row_length_, q_vecs.GetColumnPtr(q),
		  linear_transformed_q_vecs.GetColumnPtr(q));
	if(loo_alpha_vec != NULL) {
	  // For the linear system (B^T W(q) B)^{-1} B^T W(q) Y (the
	  // leave-one-out version).
	  (*loo_alpha_vec)[q] =
	    la::Dot(row_length_, loo_q_vecs->GetColumnPtr(q),
		    linear_transformed_loo_q_vecs->GetColumnPtr(q));
	}
	// For the linear system (B^T W(q) B)^{-1} t(q).
	expansion_alpha_vec[q] =
	  la::Dot(row_length_, expansion_q_vecs.GetColumnPtr(q),
		  linear_transformed_expansion_q_vecs.GetColumnPtr(q));

	// Compute the current residuals: r_j = A q_j - alpha_j q_j -
	// beta_{j-1} q_{j-1}
	//
	// For the linear system (B^T W(q) B)^{-1} B^T W(q) Y.
	la::ScaleOverwrite
	  (row_length_, 1, linear_transformed_q_vecs.GetColumnPtr(q),
	   residuals.GetColumnPtr(q));
	la::AddExpert(row_length_, -alpha_vec[q], q_vecs.GetColumnPtr(q),
		      residuals.GetColumnPtr(q));
	la::AddExpert(row_length_, -previous_beta_vec[q],
		      previous_q_vecs.GetColumnPtr(q),
		      residuals.GetColumnPtr(q));
	if(loo_alpha_vec != NULL) {
	  // For the linear system (B^T W(q) B)^{-1} B^T W(q) Y (the
	  // leave-one-out version).
	  la::ScaleOverwrite
	    (row_length_, 1, linear_transformed_loo_q_vecs->GetColumnPtr(q),
	     loo_residuals->GetColumnPtr(q));
	  la::AddExpert(row_length_, -(*loo_alpha_vec)[q], 
			loo_q_vecs->GetColumnPtr(q),
			loo_residuals->GetColumnPtr(q));
	  la::AddExpert(row_length_, -(*previous_loo_beta_vec)[q],
			previous_loo_q_vecs->GetColumnPtr(q),
			loo_residuals->GetColumnPtr(q));
	}
	// For the linear system (B^T W(q) B)^{-1} t(q).
	la::ScaleOverwrite
	  (row_length_, 1, linear_transformed_expansion_q_vecs.GetColumnPtr(q),
	   expansion_residuals.GetColumnPtr(q));
	la::AddExpert(row_length_, -expansion_alpha_vec[q], 
		      expansion_q_vecs.GetColumnPtr(q),
		      expansion_residuals.GetColumnPtr(q));
	la::AddExpert(row_length_, -previous_expansion_beta_vec[q],
		      previous_expansion_q_vecs.GetColumnPtr(q),
		      expansion_residuals.GetColumnPtr(q));

	// Compute for the current residuals: beta_j = || r_j ||.
	//
	// For the linear system (B^T W(q) B)^{-1} B^T W(q) Y.
	beta_vec[q] = 
	  la::LengthEuclidean(row_length_, residuals.GetColumnPtr(q));
	if(loo_alpha_vec != NULL) {
	  (*loo_beta_vec)[q] =
	    la::LengthEuclidean(row_length_, loo_residuals->GetColumnPtr(q));
	}
	expansion_beta_vec[q] =
	  la::LengthEuclidean(row_length_, 
			      expansion_residuals.GetColumnPtr(q));

	// Detect break-down condition
	if(beta_vec[q] < DBL_EPSILON || expansion_beta_vec[q] < DBL_EPSILON ||
	   (loo_beta_vec != NULL && (*loo_beta_vec)[q] < DBL_EPSILON)) {
	  query_in_cg_loop[q] = false;
	  num_queries_in_lanczos_loop--;
	  continue;
	}
	
	if(iter == 1) {

	  // delta_1 = alpha_1
	  delta_vec[q] = alpha_vec[q];
	  if(loo_alpha_vec != NULL) {
	    (*loo_delta_vec)[q] = (*loo_alpha_vec)[q];
	  }
	  expansion_delta_vec[q] = expansion_alpha_vec[q];

	  // q_bar_1 = q_1
	  la::ScaleOverwrite(row_length_, 1, q_vecs.GetColumnPtr(q),
			     q_bar_vecs.GetColumnPtr(q));
	  if(loo_alpha_vec != NULL) {
	    la::ScaleOverwrite(row_length_, 1, loo_q_vecs->GetColumnPtr(q),
			       loo_q_bar_vecs->GetColumnPtr(q));
	  }
	  la::ScaleOverwrite(row_length_, 1, expansion_q_vecs.GetColumnPtr(q),
			     expansion_q_bar_vecs.GetColumnPtr(q));

	  // c_1 = beta_0 / alpha_1
	  c_vec[q] = previous_beta_vec[q] / alpha_vec[q];
	  if(loo_alpha_vec != NULL) {
	    (*loo_c_vec)[q] = (*previous_loo_beta_vec)[q] / 
	      (*loo_alpha_vec)[q];
	  }
	  expansion_c_vec[q] = previous_expansion_beta_vec[q] /
	    expansion_alpha_vec[q];

	  // x_1 = c_1 q_bar_1
	  la::ScaleOverwrite(row_length_, c_vec[q], q_bar_vecs.GetColumnPtr(q),
			     solutions.GetColumnPtr(q));
	  if(loo_alpha_vec != NULL) {
	    la::ScaleOverwrite(row_length_, (*loo_c_vec)[q],
			       loo_q_bar_vecs->GetColumnPtr(q),
			       loo_solutions->GetColumnPtr(q));
	  }
	  la::ScaleOverwrite(row_length_, expansion_c_vec[q],
			     expansion_q_bar_vecs.GetColumnPtr(q),
			     query_expansion_solutions.GetColumnPtr(q));
	  
	}
	else {

	  // gamma_{j-1} = beta_{j-1} / delta_{j-1}
	  double previous_gamma_vec = previous_beta_vec[q] /
	    previous_delta_vec[q];
	  double previous_loo_gamma_vec = 0;
	  if(loo_alpha_vec != NULL) {
	    previous_loo_gamma_vec = (*previous_loo_beta_vec)[q] /
	      (*previous_loo_delta_vec)[q];
	  }
	  double previous_expansion_gamma_vec = 
	    previous_expansion_beta_vec[q] / previous_expansion_delta_vec[q];
	  
	  // delta_j = alpha_j - beta_{j-1} gamma_{j-1}
	  delta_vec[q] = alpha_vec[q] - previous_beta_vec[q] *
	    previous_gamma_vec;
	  if(loo_alpha_vec != NULL) {
	    (*loo_delta_vec)[q] = (*loo_alpha_vec)[q] - 
	      (*previous_loo_beta_vec)[q] * previous_loo_gamma_vec;
	  }
	  expansion_delta_vec[q] = expansion_alpha_vec[q] -
	    previous_expansion_beta_vec[q] * previous_expansion_gamma_vec;

	  // q_bar_j = q_j - gamma_{j-1} q_bar_{j-1}
	  la::ScaleOverwrite(row_length_, 1, q_vecs.GetColumnPtr(q),
			     q_bar_vecs.GetColumnPtr(q));
	  la::AddExpert(row_length_, -previous_gamma_vec,
			previous_q_bar_vecs.GetColumnPtr(q),
			q_bar_vecs.GetColumnPtr(q));
	  if(loo_alpha_vec != NULL) {
	    la::ScaleOverwrite(row_length_, 1, loo_q_vecs->GetColumnPtr(q),
			       loo_q_bar_vecs->GetColumnPtr(q));
	    la::AddExpert(row_length_, -previous_loo_gamma_vec,
			  previous_loo_q_bar_vecs->GetColumnPtr(q),
			  loo_q_bar_vecs->GetColumnPtr(q));
	  }
	  la::ScaleOverwrite(row_length_, 1, expansion_q_vecs.GetColumnPtr(q),
			     expansion_q_bar_vecs.GetColumnPtr(q));
	  la::AddExpert(row_length_, -previous_expansion_gamma_vec,
			previous_expansion_q_bar_vecs.GetColumnPtr(q),
			expansion_q_bar_vecs.GetColumnPtr(q));

	  // c_j = q_j^T b - gamma_{j-1} delta_{j-1} c_{j-1} / delta_j
	  c_vec[q] = la::Dot(row_length_, q_vecs.GetColumnPtr(q),
			     right_hand_sides.GetColumnPtr(q)) -
	    previous_gamma_vec * previous_delta_vec[q] *
	    previous_c_vec[q] / delta_vec[q];
	  if(loo_alpha_vec != NULL) {
	    (*loo_c_vec)[q] = la::Dot(row_length_, loo_q_vecs->GetColumnPtr(q),
				      loo_right_hand_sides->GetColumnPtr(q)) -
	      previous_loo_gamma_vec * (*previous_loo_delta_vec)[q] *
	      (*previous_loo_c_vec)[q] / (*loo_delta_vec)[q];
	  }
	  expansion_c_vec[q] = 
	    la::Dot(row_length_, expansion_q_vecs.GetColumnPtr(q),
		    query_expansions.GetColumnPtr(q)) -
	    previous_expansion_gamma_vec * previous_expansion_delta_vec[q] *
	    previous_expansion_c_vec[q] / expansion_delta_vec[q];

	  // Update solutions: x_j = x_{j-1} + c_j q_bar_j
	  la::AddExpert(row_length_, c_vec[q], q_bar_vecs.GetColumnPtr(q),
			solutions.GetColumnPtr(q));
	  if(loo_alpha_vec != NULL) {
	    la::AddExpert(row_length_, (*loo_c_vec)[q], 
			  loo_q_bar_vecs->GetColumnPtr(q),
			  loo_solutions->GetColumnPtr(q));
	  }
	  la::AddExpert(row_length_, expansion_c_vec[q], 
			expansion_q_bar_vecs.GetColumnPtr(q),
			query_expansion_solutions.GetColumnPtr(q));
	}

      } // end of iterating over each query point.
      
      // Here we need to copy over the current stuff to the previous
      // stuffs...
      
      // Copy beta's
      previous_beta_vec.CopyValues(beta_vec);
      previous_expansion_beta_vec.CopyValues(expansion_beta_vec);
      
      // Copy q_vec's
      previous_q_vecs.CopyValues(q_vecs);
      previous_expansion_q_vecs.CopyValues(expansion_q_vecs);
      
      // Copy delta's
      previous_delta_vec.CopyValues(delta_vec);
      previous_expansion_delta_vec.CopyValues(expansion_delta_vec);
      
      // Copy qbar's
      previous_q_bar_vecs.CopyValues(q_bar_vecs);
      previous_expansion_q_bar_vecs.CopyValues(expansion_q_bar_vecs);
      
      // Copy residuals
      previous_residuals.CopyValues(residuals);
      previous_expansion_residuals.CopyValues(expansion_residuals);
      
      // Copy c's
      previous_c_vec.CopyValues(c_vec);
      previous_expansion_c_vec.CopyValues(expansion_c_vec);
      
      if(loo_alpha_vec != NULL) {
	
	// Copy beta's
	previous_loo_beta_vec->CopyValues(*loo_beta_vec);
	
	// Copy q_vec's
	previous_loo_q_vecs->CopyValues(*loo_q_vecs);
	
	// Copy delta's
	previous_loo_delta_vec->CopyValues(*loo_delta_vec);
	
	// Copy qbar's
	previous_loo_q_bar_vecs->CopyValues(*loo_q_bar_vecs);
	
	// Copy residuals
	previous_loo_residuals->CopyValues(*loo_residuals);
	
	// Copy c's
	previous_loo_c_vec->CopyValues(*loo_c_vec);
      }

    } // end of iterating over each iteration of Lanczos.

    // I have to clean up the memory here...
    if(loo_alpha_vec != NULL) {
      delete loo_beta_vec;
      delete previous_loo_beta_vec;
      delete loo_q_vecs;
      delete previous_loo_q_vecs;
      delete loo_delta_vec;
      delete previous_loo_delta_vec;
      delete loo_q_bar_vecs;
      delete previous_loo_q_bar_vecs;
      delete loo_residuals;
      delete previous_loo_residuals;
      delete loo_c_vec;
      delete previous_loo_c_vec;
    }
  }

};

#endif
