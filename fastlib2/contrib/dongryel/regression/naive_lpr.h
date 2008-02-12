/** @file naive_lpr.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *
 *  @bug No known bugs.
 */

#ifndef NAIVE_LPR_H
#define NAIVE_LPR_H

#include "matrix_util.h"
#include "multi_index_util.h"
#include "fastlib/fastlib.h"

template<typename TKernel, int lpr_order = 1>
class NaiveLpr {

  FORBID_ACCIDENTAL_COPIES(NaiveLpr);
  
 private:
  
  /** @brief The module holding the parameters necessary for execution.
   */
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

  /** @brief The total number of coefficients for the local
   *         polynomial.
   */
  int total_num_coeffs_;

  /** @brief The dimensionality.
   */
  int dimension_;

 public:
  
  /** @brief The constructor which does nothing.
   */
  NaiveLpr() {}

  /** @brief The destructor which does nothing.
   */
  ~NaiveLpr() {}

  /** @brief Compute the local polynomial regression values using the
   *         brute-force algorithm.
   */
  void Compute() {

    // Temporary variable for storing multivariate expansion of a
    // reference point.
    Vector reference_point_expansion;
    reference_point_expansion.Init(total_num_coeffs_);

    printf("\nStarting naive local polynomial of order %d...\n", lpr_order);
    fx_timer_start(NULL, "naive_local_linear_compute");

    // Compute unnormalized sum for the numerator vector and the
    // denominator matrix.
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      // Get the query point.
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {

	// Get the reference point and the reference target training
	// value.
	const double *r_col = rset_.GetColumnPtr(r);
	const double r_target = rset_targets_[r];

	// Compute the reference point expansion.
	MultiIndexUtil::ComputePointMultivariatePolynomial
	  (dimension_, lpr_order, r_col, reference_point_expansion.ptr());
	
	// Compute the pairwise distance and the resulting kernel value.
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	double kernel_value = kernel_.EvalUnnormOnSq(dsqd);

	for(index_t i = 0; i < total_num_coeffs_; i++) {

	  numerator_[q][i] += r_target * kernel_value * 
	    reference_point_expansion[i];
	  
	  // Here, compute each component of the denominator matrix.
	  for(index_t j = 0; j < total_num_coeffs_; j++) {
	    denominator_[q].set(j, i, denominator_[q].get(j, i) +
				reference_point_expansion[j] * 
				reference_point_expansion[i] * kernel_value);
	  } // End of looping over each (j, i)-th component of the
	    // denominator matrix.
	} // End of looping over each i-th component of the numerator
	  // vector.

      } // end of looping over each reference point
    } // end of looping over each query point

    Matrix denominator_inv_q;
    denominator_inv_q.Init(total_num_coeffs_, total_num_coeffs_);
    
    // now iterate over all query points and compute regression estimate
    for(index_t q = 0; q < qset_.n_cols(); q++) {

      const double *q_col = qset_.GetColumnPtr(q);
      Vector beta_q;
      
      // Now invert the denominator matrix for each query point and
      // multiply by the numerator vector.
      MatrixUtil::PseudoInverse(denominator_[q], &denominator_inv_q);      
      la::MulInit(denominator_inv_q, numerator_[q], &beta_q);

      // Compute the dot product between the multiindex vector for the
      // query point by the beta_q.
      regression_values_[q] = beta_q[0];
      for(index_t i = 1; i <= qset_.n_rows(); i++) {
	regression_values_[q] += beta_q[i] * q_col[i - 1];
      }
    }
    
    fx_timer_stop(NULL, "naive_local_linear_compute");
    printf("\nNaive local polynomial of order %d completed...\n", lpr_order);
  }

  /** @brief Initialize the naive algorithm for initial usage.
   *
   *  @param queries The column-oriented query dataset.
   *  @param refererences The column-oriented reference dataset.
   *  @param reference_targets The training values for the reference set.
   *  @param module_in The module holding the parameters necessary for
   *                   execution.
   */
  void Init(Matrix &queries, Matrix &references, Matrix &reference_targets,
	    struct datanode *module_in) {

    // Set the module to the incoming one.
    module_ = module_in;

    // Set the dimensionality.
    dimension_ = queries.n_rows();

    // Copy the datasets.
    qset_.Copy(queries);
    rset_.Copy(references);

    // Read the reference weights.
    rset_targets_.Copy(reference_targets.GetColumnPtr(0),
		       reference_targets.n_cols());
    
    // Get bandwidth.
    kernel_.Init(fx_param_double_req(module_, "bandwidth"));

    // Compute total number of coefficients.
    total_num_coeffs_ = (int) 
      math::BinomialCoefficient(lpr_order + qset_.n_rows(), qset_.n_rows());
    
    // Allocate temporary storages for storing the numerator vectors
    // and the denominator matrices.
    numerator_.Init(qset_.n_cols());
    denominator_.Init(qset_.n_cols());

    for(index_t i = 0; i < qset_.n_cols(); i++) {
      numerator_[i].Init(total_num_coeffs_);
      numerator_[i].SetZero();
      denominator_[i].Init(total_num_coeffs_, total_num_coeffs_);
      denominator_[i].SetZero();
    }
    
    // Allocate the space for holding the computed regression values.
    regression_values_.Init(qset_.n_cols());
    regression_values_.SetZero();
  }

  /** @brief Output the naive results to the file or the screen.
   */
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
