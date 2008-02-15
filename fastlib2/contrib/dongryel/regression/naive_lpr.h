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

template<typename TKernel>
class NaiveLpr {

  FORBID_ACCIDENTAL_COPIES(NaiveLpr);
  
 private:

  ////////// Parameter related variables //////////

  /** @brief The module holding the parameters necessary for
   *         execution.
   */
  struct datanode *module_;

  /** @brief The local polynomial approximation order.
   */
  int lpr_order_;

  /** @brief The total number of coefficients for the local
   *         polynomial.
   */
  int total_num_coeffs_;

  /** @brief The dimensionality.
   */
  int dimension_;

  /** @brief The kernel function.
   */
  TKernel kernel_;

  /** @brief The Z score to use for confidence bands.
   */
  double z_score_;

  ////////// Datasets //////////

  /** @brief The column-oriented reference dataset.
   */
  Matrix rset_;

  /** @brief The reference target training values.
   */
  Vector rset_targets_;

  ////////// Computed during the training phase //////////

  /** @brief The computed fit values at each reference point.
   */
  Vector rset_regression_estimates_;
  
  /** @brief The confidence band on the fit at each reference point.
   */
  ArrayList<DRange> rset_confidence_bands_;

  /** @brief The influence value at each reference point.
   */
  Vector rset_influence_values_;

  /** @brief The magnitude of the weight diagram vector at each
   *         reference point.
   */
  Vector rset_magnitude_weight_diagrams_;

  /** @brief The first degree of freedom, i.e. the sum of the
   *         influence value at each reference point.
   */
  double rset_first_degree_of_freedom_;
  
  /** @brief The second degree of freedom, i.e. the sum of the
   *         magnitudes of the weight diagram at each reference point.
   */
  double rset_second_degree_of_freedom_;

  /** @brief The variance of the reference set.
   */
  double rset_variance_;

  ////////// Private Member Functions //////////

  /** @brief Compute the local polynomial regression values using the
   *         brute-force algorithm.
   */
  void BasicCompute_(const Matrix &queries, Vector *query_regression_estimates,
		     Vector *query_magnitude_weight_diagrams, 
		     Vector *query_influence_values, bool leave_one_out) {

    // Allocate memory to hold the final results.
    query_regression_estimates->Init(queries.n_cols());
    query_magnitude_weight_diagrams->Init(queries.n_cols());

    if(query_influence_values != NULL) {
      query_influence_values->Init(queries.n_cols());
    }

    // Temporary variables to hold intermediate computation results.
    ArrayList<Matrix> denominator;
    denominator.Init(queries.n_cols());
    Matrix numerator;
    numerator.Init(total_num_coeffs_, queries.n_cols());
    numerator.SetZero();
    ArrayList<Matrix> weight_diagram_numerator;
    weight_diagram_numerator.Init(queries.n_cols());
    for(index_t i = 0; i < queries.n_cols(); i++) {
      denominator[i].Init(total_num_coeffs_, total_num_coeffs_);
      denominator[i].SetZero();
      weight_diagram_numerator[i].Init(total_num_coeffs_, total_num_coeffs_);
      weight_diagram_numerator[i].SetZero();
    }
    
    // Temporary variable for storing multivariate expansion of a
    // point.
    Vector point_expansion;
    point_expansion.Init(total_num_coeffs_);

    // Compute unnormalized sum for the numerator vector and the
    // denominator matrix.
    for(index_t q = 0; q < queries.n_cols(); q++) {
      
      // Get the query point.
      const double *q_col = queries.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {

	// If leave one out is toggled on, then we skip the index with
	// the same number.
	if(leave_one_out && q == r) {
	  continue;
	}

	// Get the reference point and the reference target training
	// value.
	const double *r_col = rset_.GetColumnPtr(r);
	const double r_target = rset_targets_[r];

	// Compute the reference point expansion.
	MultiIndexUtil::ComputePointMultivariatePolynomial
	  (dimension_, lpr_order_, r_col, point_expansion.ptr());
	
	// Compute the pairwise distance and the resulting kernel value.
	double dsqd = la::DistanceSqEuclidean(queries.n_rows(), q_col, r_col);
	double kernel_value = kernel_.EvalUnnormOnSq(dsqd);

	for(index_t i = 0; i < total_num_coeffs_; i++) {

	  // Compute each component of the numerator matrix.
	  numerator.set(i, q, numerator.get(i, q) +
			r_target * kernel_value * point_expansion[i]);
	  
	  // Here, compute each component of the denominator matrix.
	  for(index_t j = 0; j < total_num_coeffs_; j++) {
	    denominator[q].set(j, i, denominator[q].get(j, i) +
			       point_expansion[j] * point_expansion[i] * 
			       kernel_value);
	    weight_diagram_numerator[q].set
	      (j, i, weight_diagram_numerator[q].get(j, i) +
	       point_expansion[j] * point_expansion[i] * kernel_value *
	       kernel_value);
	    
	  } // End of looping over each (j, i)-th component of the
	    // denominator matrix.
	} // End of looping over each i-th component of the numerator
	  // vector.

      } // end of looping over each reference point
    } // end of looping over each query point

    Matrix denominator_inv_q;
    denominator_inv_q.Init(total_num_coeffs_, total_num_coeffs_);
    
    // now iterate over all query points and compute regression estimate
    for(index_t q = 0; q < queries.n_cols(); q++) {

      const double *q_col = queries.GetColumnPtr(q);
      Vector beta_q;
      Vector q_numerator;
      numerator.MakeColumnVector(q, &q_numerator);
      
      // Compute the query point expansion.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, q_col, point_expansion.ptr());

      // Now invert the denominator matrix for each query point and
      // multiply by the numerator vector.
      MatrixUtil::PseudoInverse(denominator[q], &denominator_inv_q);      
      la::MulInit(denominator_inv_q, q_numerator, &beta_q);

      // Compute the dot product between the multiindex vector for the
      // query point by the beta_q.
      (*query_regression_estimates)[q] = la::Dot(beta_q, point_expansion);

      // Now we compute the magnitude of the weight diagram for each
      // query point.
      Vector pseudo_inverse_times_query_expansion;
      Vector intermediate_product;
      la::MulInit(denominator_inv_q, point_expansion,
		  &pseudo_inverse_times_query_expansion);
      la::MulInit(weight_diagram_numerator[q],
		  pseudo_inverse_times_query_expansion, &intermediate_product);
      (*query_magnitude_weight_diagrams)[q] =
	sqrt(la::Dot(pseudo_inverse_times_query_expansion, 
		     intermediate_product));

      // Compute the influence value at each point (if it belongs to
      // the reference set), i.e. (r(q))^T (B^T W(q) B)^-1 B^T W(q)
      // e_i = (r(q))^T (B^T W(q) B)-1 r(q).
      if(query_influence_values != NULL) {
	(*query_influence_values)[q] =
	  la::Dot(point_expansion, pseudo_inverse_times_query_expansion);
      }      
    } // end of iterating over each query point.
  }

  void ComputeConfidenceBands_(const Matrix &queries,
			       Vector *query_regression_estimates,
			       ArrayList<DRange> *query_confidence_bands,
			       Vector *query_magnitude_weight_diagrams) {

    // Initialize the storage for the confidene bands.
    query_confidence_bands->Init(queries.n_cols());
    
    for(index_t q = 0; q < queries.n_cols(); q++) {
      DRange &q_confidence_band = (*query_confidence_bands)[q];
      double spread = z_score_ * (*query_magnitude_weight_diagrams)[q] * 
	sqrt(rset_variance_);

      q_confidence_band.lo = (*query_regression_estimates)[q] - spread;
      q_confidence_band.hi = (*query_regression_estimates)[q] + spread;
    }
  }

  /** @brief Computes the variance by the normalized redisual sum of
   *         squares for the reference dataset.
   */
  void ComputeVariance_() {

    // Compute the degrees of freedom, i.e. the sum of the influence
    // values at each reference point and the sum of the squared
    // magnitudes of the weight diagram vectors at each reference
    // point.
    rset_first_degree_of_freedom_ = rset_second_degree_of_freedom_ = 0;
    for(index_t i = 0; i < rset_.n_cols(); i++) {
      rset_first_degree_of_freedom_ += rset_influence_values_[i];
      rset_second_degree_of_freedom_ += rset_magnitude_weight_diagrams_[i] * 
	rset_magnitude_weight_diagrams_[i];
    }

    // Reset the sum accumulated to zero.
    rset_variance_ = 0;

    // Loop over each reference point and add up the residual.
    for(index_t i = 0; i < rset_.n_cols(); i++) {
      double prediction_error = rset_targets_[i] - 
	rset_regression_estimates_[i];
      rset_variance_ += prediction_error * prediction_error;
    }
    
    // This could happen if enough matrices are singular...
    if(rset_.n_cols() - 2.0 * rset_first_degree_of_freedom_ +
       rset_second_degree_of_freedom_ <= 0) {
      rset_variance_ = DBL_MAX;
    }

    rset_variance_ *= 1.0 / 
      (rset_.n_cols() - 2.0 * rset_first_degree_of_freedom_ +
       rset_second_degree_of_freedom_);
  }

  /** @brief Predicts the regression estimates along with the
   *         confidence intervals for the given set of query points.
   */
  void ComputeMain_(const Matrix &queries, Vector *query_regression_estimates,
		    ArrayList<DRange> *query_confidence_bands,
		    Vector *query_magnitude_weight_diagrams,
		    Vector *query_influence_values, bool leave_one_out) {

    BasicCompute_(queries, query_regression_estimates,
		  query_magnitude_weight_diagrams, query_influence_values,
		  leave_one_out);

    // If the reference dataset is being used for training, then
    // compute variance and degrees of freedom.
    if(query_influence_values != NULL) {
      ComputeVariance_();
    }

    ComputeConfidenceBands_(queries, query_regression_estimates,
			    query_confidence_bands,
			    query_magnitude_weight_diagrams);
  }
  
 public:

  ////////// Getter/Setters //////////
  
  /** @brief Get the regression estimates of the model (i.e. on the
   *         reference set).
   *
   *  @param results The uninitialized vector which will be filled
   *                 with the computed regression estimates.
   */
  void get_model_estimates(Vector *rset_regression_estimates_copy,
			   ArrayList<DRange> *rset_confidence_bands_copy,
			   Vector *rset_magnitude_weight_diagrams_copy,
			   Vector *rset_influence_values_copy,
			   double *rset_first_degree_of_freedom_copy,
			   double *rset_second_degree_of_freedom_copy,
			   double *rset_variance_copy) {

    rset_regression_estimates_copy->Copy(rset_regression_estimates_);
    rset_confidence_bands_copy->Copy(rset_confidence_bands_);
    rset_magnitude_weight_diagrams_copy->Copy(rset_magnitude_weight_diagrams_);
    rset_influence_values_copy->Copy(rset_influence_values_);
    *rset_first_degree_of_freedom_copy = rset_first_degree_of_freedom_;
    *rset_second_degree_of_freedom_copy = rset_second_degree_of_freedom_;
    *rset_variance_copy = rset_variance_;
  }

  ////////// Constructor/Destructor //////////

  /** @brief The constructor which does nothing.
   */
  NaiveLpr() {}

  /** @brief The destructor which does nothing.
   */
  ~NaiveLpr() {}

  ////////// User-level Functions //////////
  
  void Compute(const Matrix &queries, Vector *query_regression_estimates,
	       ArrayList<DRange> *query_confidence_bands,
	       Vector *query_magnitude_weight_diagrams,
	       Vector *query_influence_values) {

    ComputeMain_(queries, query_regression_estimates, query_confidence_bands, 
		 query_magnitude_weight_diagrams, query_influence_values, 
		 false);
  }

  /** @brief Initialize the naive algorithm for initial usage,
   *         i.e. the training phase.
   *
   *  @param references The column-oriented reference dataset.
   *  @param reference_targets The training values for the reference set.
   *  @param module_in The module holding the parameters necessary for
   *                   execution.
   */
  void Init(Matrix &references, Matrix &reference_targets,
	    struct datanode *module_in) {

    // Set the module to the incoming one.
    module_ = module_in;

    // Set the local polynomial order.
    lpr_order_ = fx_param_int(module_in, "lpr_order", 0);

    // Set the z-score
    z_score_ = fx_param_double(module_in, "z_score", 1.96);

    // Set the dimensionality.
    dimension_ = references.n_rows();

    // Copy the datasets and the reference target training values.
    rset_.Copy(references);
    rset_targets_.Copy(reference_targets.GetColumnPtr(0),
		       reference_targets.n_cols());
    
    // Get bandwidth.
    kernel_.Init(fx_param_double_req(module_, "bandwidth"));

    // Compute total number of coefficients.
    total_num_coeffs_ = (int) 
      math::BinomialCoefficient(lpr_order_ + rset_.n_rows(), rset_.n_rows());

    // Train the model using the reference set (i.e. compute
    // confidence interval and degrees of freedom.)
    ComputeMain_(references, &rset_regression_estimates_, 
		 &rset_confidence_bands_, &rset_magnitude_weight_diagrams_, 
		 &rset_influence_values_, true);
  }

};

#endif
