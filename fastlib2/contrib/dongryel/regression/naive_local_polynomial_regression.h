#ifndef NAIVE_LOCAL_POLYNOMIAL_REGRESSION_H
#define NAIVE_LOCAL_POLYNOMIAL_REGRESSION_H

#include "fastlib/fastlib_int.h"

template<typename TKernel, int order = 1>
class NaiveLocalPolynomialRegression {

  FORBID_ACCIDENTAL_COPIES(NaiveLocalPolynomialRegression);

 private:

  struct datanode *module_;

  /** @brief The column-oriented query dataset.
   */
  Matrix qset_;

  /** @brief The column-oriented reference dataset.
   */
  Matrix rset_;

  /** @brief The reference target training values.
   */
  Vector rset_targets_;

  /** @brief The kernel function.
   */
  TKernel kernel_;

  /** @brief The numerator vector X^T W(q) Y for each query point.
   */
  ArrayList<Vector> numerator_;
  
  /** @brief The denominator matrix X^T W(q) X for each query point.
   */
  ArrayList<Matrix> denominator_;

  /** @brief The computed regression values.
   */
  Vector regression_values_;

  /** total number of coefficients for the local polynomial */
  int total_num_coeffs_;

  void PseudoInverse(const Matrix &A, Matrix *A_inv) {
    Vector ro_s;
    Matrix ro_U, ro_VT;

    // compute the SVD of A
    la::SVDInit(A, &ro_s, &ro_U, &ro_VT);
    
    // take the transpose of V^T and U
    Matrix ro_VT_trans;
    Matrix ro_U_trans;
    la::TransposeInit(ro_VT, &ro_VT_trans);
    la::TransposeInit(ro_U, &ro_U_trans);
    Matrix ro_s_inv;
    ro_s_inv.Init(ro_VT_trans.n_cols(), ro_U_trans.n_rows());
    ro_s_inv.SetZero();
    
    printf("Condition number: %g / %g = %g\n", ro_s[0],
	   ro_s[ro_s.length() - 1], ro_s[0] / ro_s[ro_s.length() - 1]);

    // initialize the diagonal by the inverse of ro_s
    for(index_t i = 0; i < ro_s.length(); i++) {
      if(ro_s[i] > 0.001 * ro_s[0]) {
	ro_s_inv.set(i, i, 1.0 / ro_s[i]);
      }
      else {
	ro_s_inv.set(i, i, 0);
      }
    }
    Matrix intermediate;
    la::MulInit(ro_s_inv, ro_U_trans, &intermediate);
    la::MulInit(ro_VT_trans, intermediate, A_inv);
  }

 public:
  
  NaiveLocalPolynomialRegression() {}

  ~NaiveLocalPolynomialRegression() {}

  void Compute() {

    printf("\nStarting naive local polynomial of order %d...\n", order);
    fx_timer_start(NULL, "naive_local_linear_compute");

    // compute unnormalized sum for the numerator vector and the denominator
    // matrix
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {

	const double *r_col = rset_.GetColumnPtr(r);
	const double r_target = rset_targets_[r];
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double kernel_value = kernel_.EvalUnnormOnSq(dsqd);

	for(index_t i = 0; i <= qset_.n_rows(); i++) {

	  double factor_i;

	  // Compute the numerator vector.
	  if(i == 0) {
	    numerator_[q][0] += r_target * kernel_value;
	    factor_i = 1.0;
	  }
	  else {
	    numerator_[q][i] += r_col[i - 1] * r_target * kernel_value;
	    factor_i = r_col[i - 1];
	  }
	  
	  // Here, compute each component of the denominator matrix.
	  for(index_t j = 0; j <= qset_.n_rows(); j++) {
	    double factor_j;

	    if(j == 0) {
	      factor_j = 1.0;	      
	    }
	    else {
	      factor_j = r_col[j - 1];
	    }

	    denominator_[q].set(j, i, denominator_[q].get(j, i) +
				factor_j * factor_i * kernel_value);
	  }
	}

      } // end of looping over each reference point
    } // end of looping over each query point

    // now iterate over all query points and compute regression estimate
    for(index_t q = 0; q < qset_.n_cols(); q++) {

      const double *q_col = qset_.GetColumnPtr(q);
      Matrix denominator_inv_q;
      Vector beta_q;
      
      // now invert the denominator matrix for each query point and multiply
      // by the numerator vector
      PseudoInverse(denominator_[q], &denominator_inv_q);      
      la::MulInit(denominator_inv_q, numerator_[q], &beta_q);

      // compute the dot product between the multiindex vector for the query
      // point by the beta_q
      regression_values_[q] = beta_q[0];
      for(index_t i = 1; i <= qset_.n_rows(); i++) {
	regression_values_[q] += beta_q[i] * q_col[i - 1];
      }
    }
    
    fx_timer_stop(NULL, "naive_local_linear_compute");
    printf("\nNaive local polynomial of order %d completed...\n", order);
  }

  void Init(Matrix &queries, Matrix &references, Matrix &reference_targets,
	    struct datanode *module_in) {

    // Set the module to the incoming.
    module_ = module_in;

    // Copy the datasets.
    qset_.Copy(queries);
    rset_.Copy(references);

    // read the reference weights
    
    rset_targets_.Copy(reference_targets.GetColumnPtr(0),
		       reference_targets.n_cols());

    // get bandwidth
    kernel_.Init(fx_param_double_req(module_, "bandwidth"));

    // compute total number of coefficients
    total_num_coeffs_ = (int) 
      math::BinomialCoefficient(order + qset_.n_rows(), qset_.n_rows());
    
    // allocate temporary storages for storing the numerator vectors and
    // the denominator matrices
    numerator_.Init(qset_.n_cols());
    denominator_.Init(qset_.n_cols());

    for(index_t i = 0; i < qset_.n_cols(); i++) {
      numerator_[i].Init(total_num_coeffs_);
      numerator_[i].SetZero();
      denominator_[i].Init(total_num_coeffs_, total_num_coeffs_);
      denominator_[i].SetZero();
    }
    
    // allocate density storage
    regression_values_.Init(qset_.n_cols());
    regression_values_.SetZero();
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;

    if((fname = fx_param_str(module_, 
			     "naive_local_polynomial_output", 
			     "naive_local_polynomial_output.txt")) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      fprintf(stream, "%g\n", regression_values_[q]);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }    
  }

  void ComputeMaximumRelativeError(const Vector &approx_estimate) {
    
    double max_rel_err = 0;
    for(index_t q = 0; q < regression_values_.length(); q++) {
      double rel_err = fabs(approx_estimate[q] - regression_values_[q]) / 
	regression_values_[q];
      
      if(rel_err > max_rel_err) {
	max_rel_err = rel_err;
      }
    }
    
    fx_format_result(NULL, "maxium_relative_error_for_fast_LPR", "%g", 
		     max_rel_err);
  }

};

#endif
