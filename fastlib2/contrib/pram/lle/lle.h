/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file lle.h
 *
 * Defines a Local Linear Embedding object 
 * and performs nonlinear dimensionality reduction 
 * of a given manifold in a higher dimension
 * by locally linear embedding.
 * 
 * This implements the Local Linear Embedding 
 * described in the following paper:
 * @article{roweis00nonlinear,
 *   author = "S. Roweis and L. Saul",
 *   title = "Nonlinear dimensionality reduction by locally linear embedding",
 *   journal = "Science",
 *   year = "2000"
 * }
 *
 */

#ifndef LLE_H
#define LLE_H

#include <fastlib/fastlib.h>

// this path is already included in the build pathname, 
// so no change is required in the file pathname
#include "mlpack/allknn/allknn.h"

// changed as per review
#define LEAF_LENGTH 20

/**
 * This is a Local Linear Embedding class
 *
 * It computes the lower dimensional embedding of 
 * a manifold in higher dimension
 *
 * Example use:
 *
 * @code
 * LLE lle;
 * Matrix dataset;
 * index_t knn;
 * datanode *lle_module;
 * ...........
 * fx_param_int(lle_module,"knn",knn);
 * lle.Init(dataset, lle_module);
 * lle.Compute();
 *
 * Matrix lower_dimensional_embeddings;
 * lle.OutputResults(&lower_dimensional_embeddings);
 * @endcode
 *
 */
class LLE {

 private:

  // The original data matrix and the lower dimensional embeddings
  Matrix x_mat_, embeddings_mat_;

  // The module for the object
  datanode *lle_module_;

  // A helper function which computes the trace of a square 
  // matrix and terminates the program if the matrix is 
  // not square
  double Trace_(const Matrix& mat) {

    DEBUG_ASSERT_MSG(mat.n_cols() == mat.n_rows(),
		     "Trace: matrix must be square\n");
    index_t n = mat.n_cols();
    double trace = 0.0;

    for (index_t i = 0; i < n; i++) {
      trace += mat.get(i,i);
    }

    return trace;
  }

 public:

  // The basic constructor function
  LLE() {
  }

  // The basic destructor function
  ~LLE() {
  }

  // The setter functions

  void set_x(const Matrix& mat) {
    x_mat_.Copy(mat);
    return;
  }

  void set_embeddings(const Matrix& mat) {
    embeddings_mat_.Copy(mat);
    return;
  }

  // The getter functions
  
  // As per fastlib coding style, getters are not supposed to have 
  // 'get' in the beginning of the function name....
  // For a private variable: 'index_t private_variable_;'
  // the getter function should be: 'index_t private_variable();'

  const Matrix& x() {
    return x_mat_;
  }

  const Matrix& embeddings() {
    return embeddings_mat_;
  }

  void Init(const Matrix& data, datanode *lle_module) {

    lle_module_ = lle_module;
    set_x(data);

    return;
  }

  /**
   * This function performs the nonlinear dimensionality
   * reduction on the given dataset
   */
  void Compute() {

    double tolerance;
    Vector one_vec;
    index_t n = fx_param_int_req(lle_module_, "N");
    index_t knns = fx_param_int_req(lle_module_, "knn");
    index_t dim = fx_param_int_req(lle_module_, "D");
    index_t lower_dim = fx_param_int_req(lle_module_, "d");
    Matrix weights_mat, complete_weights_mat;

    one_vec.Init(knns);
    one_vec.SetAll(1.0);
    weights_mat.Init(knns, n);
    weights_mat.SetZero();
    complete_weights_mat.Init(n, n);
    complete_weights_mat.SetZero();

    // STEP 1: Finding the K-NN of each point

    AllkNN allknn;
    ArrayList<index_t> neighbor_indices;
    ArrayList<double> dist_sq;

    NOTIFY("Finding %"LI"d nearest neighbors of %"LI"d points\n",
	   knns, n);

    // changed as per review
    allknn.Init(x_mat_, x_mat_, LEAF_LENGTH, knns+1);
    allknn.ComputeNeighbors(&neighbor_indices, &dist_sq);

    // STEP 2: Solving for reconstruction weights

    // Regularizing in the case when knns > dim
    if (knns > dim) {
      NOTIFY("knn > D, regularization used\n");
      tolerance = 1.0e-3;
    }
    else {
      tolerance = 0.0;
    }

    for (index_t i = 0; i < n; i++) {

      Matrix centered_neighbors_mat, temp_trans_mat, local_covariance_mat;
      Vector x_i_vec, w_i_vec;

      centered_neighbors_mat.Init(dim, knns);
      x_mat_.MakeColumnVector(i, &x_i_vec);

      for (index_t j = 0; j < knns; j++) {
	Vector tmp_vec;
	centered_neighbors_mat.MakeColumnVector(j, &tmp_vec);
	tmp_vec.CopyValues(x_mat_.GetColumnPtr(neighbor_indices[(knns+1) * i 
								+ j + 1]));
	la::SubFrom(x_i_vec, &tmp_vec);
      }

      la::TransposeInit(centered_neighbors_mat, &temp_trans_mat);
      la::MulInit(temp_trans_mat, centered_neighbors_mat,
		  &local_covariance_mat);

      weights_mat.MakeColumnVector(i, &w_i_vec);

      // regularizing the local covariance matrix in the case 
      // when knns > dim

      Vector temp_diag_vec;
      Matrix regularization_mat;

      temp_diag_vec.Init(knns);
      temp_diag_vec.SetAll(tolerance*Trace_(local_covariance_mat));

      regularization_mat.Init(knns, knns);
      regularization_mat.SetDiagonal(temp_diag_vec);

      la::AddTo(regularization_mat, &local_covariance_mat);

      // solving Cw = 1

      Vector temp_solve_vec;
      success_t solve_op = la::SolveInit(local_covariance_mat, one_vec,
					 &temp_solve_vec);
      DEBUG_ASSERT_MSG(solve_op == SUCCESS_PASS, "Solving Cw = 1 failed\n");
 
      // enforcing sum(w) = 1
      double total_weight = la::Dot(temp_solve_vec, one_vec);
      la::ScaleOverwrite(1.0/total_weight, temp_solve_vec, 
			 &w_i_vec);
    }

    NOTIFY("Weights computed\n");

    for (index_t i = 0; i < n; i++) {
      for (index_t j = 0; j < knns; j++) {
	complete_weights_mat.set(i, neighbor_indices[(knns+1) * i + j + 1], 
				 weights_mat.get(j, i)); 
      }
    }

    NOTIFY("Complete W matrix formed for the cost matrix\n");

    // STEP 3: Compute embedding from eigenvectors of cost matrix 
    // M = (I - W)'(I - W)

    Vector diag_vec, s_vec;
    Matrix u_mat, v_trans_mat, embeddings_trans_mat, embeddings_mat;
    Matrix eye_mat, temp_trans_mat, cost_mat, cost_trans_mat, temp_mat;
    Matrix v_mat;

    diag_vec.Init(n);
    diag_vec.SetAll(1.0);

    eye_mat.Init(n, n);
    eye_mat.SetDiagonal(diag_vec);

    la::SubFrom(complete_weights_mat, &eye_mat);
    la::TransposeInit(eye_mat,  &temp_trans_mat);
    la::MulInit(temp_trans_mat, eye_mat, &cost_mat);
    la::TransposeInit(cost_mat, &cost_trans_mat);
    la::MulInit(cost_trans_mat, cost_mat, &temp_mat);

    success_t svd_op = la::SVDInit(temp_mat, &s_vec, 
				    &u_mat, &v_trans_mat);
    DEBUG_ASSERT_MSG(svd_op == SUCCESS_PASS, "SVD(M'M) failed miserably\n");

    la::TransposeInit(v_trans_mat, &v_mat);

    u_mat.MakeColumnSlice(n - lower_dim - 1, lower_dim, 
			  &embeddings_trans_mat);
    la::Scale(sqrt(n), &embeddings_trans_mat);
    la::TransposeInit(embeddings_trans_mat, &embeddings_mat);
    set_embeddings(embeddings_mat);

    return;
  }

  /**
   * This function outputs the embeddings 
   * obtained after the nonlinear dimensionality
   * reduction.
   */
  void OutputResults(Matrix *lower_dimensional_embedding) {

    lower_dimensional_embedding->Copy(embeddings());

    return;
  }
};
#endif
