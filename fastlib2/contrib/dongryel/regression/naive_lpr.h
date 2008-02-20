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
#include "mlpack/allknn/allknn.h"

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

  /** @brief The kernel function on each reference point.
   */
  ArrayList<TKernel> kernels_;

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
		     Vector *query_influence_values) {

    // Allocate memory to hold the final results.
    query_regression_estimates->Init(queries.n_cols());
    query_magnitude_weight_diagrams->Init(queries.n_cols());

    if(query_influence_values != NULL) {
      query_influence_values->Init(queries.n_cols());
    }

    // Temporary variables to hold intermediate computation results.
    Matrix denominator;
    denominator.Init(total_num_coeffs_, total_num_coeffs_);
    Vector numerator;
    numerator.Init(total_num_coeffs_);
    Matrix weight_diagram_numerator;
    weight_diagram_numerator.Init(total_num_coeffs_, total_num_coeffs_);
    
    // Temporary variable for storing multivariate expansion of a
    // point.
    Vector point_expansion;
    point_expansion.Init(total_num_coeffs_);

    // Temporary variable for holding the pseudoinverse.
    Matrix denominator_inv_q;
    denominator_inv_q.Init(total_num_coeffs_, total_num_coeffs_);

    // Compute unnormalized sum for the numerator vector and the
    // denominator matrix.
    for(index_t q = 0; q < queries.n_cols(); q++) {

      // Initialize the temporary variables holding the sum.
      numerator.SetZero();
      denominator.SetZero();
      weight_diagram_numerator.SetZero();

      // Get the query point.
      const double *q_col = queries.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {

	// Get the reference point and the reference target training
	// value.
	const double *r_col = rset_.GetColumnPtr(r);
	const double r_target = rset_targets_[r];

	// Compute the reference point expansion.
	MultiIndexUtil::ComputePointMultivariatePolynomial
	  (dimension_, lpr_order_, r_col, point_expansion.ptr());
	
	// Compute the pairwise distance and the resulting kernel value.
	double dsqd = la::DistanceSqEuclidean(queries.n_rows(), q_col, r_col);
	double kernel_value = kernels_[r].EvalUnnormOnSq(dsqd);

	for(index_t i = 0; i < total_num_coeffs_; i++) {

	  // Compute each component of the numerator matrix.
	  numerator[i] += r_target * kernel_value * point_expansion[i];
	  
	  // Here, compute each component of the denominator matrix.
	  for(index_t j = 0; j < total_num_coeffs_; j++) {
	    denominator.set(j, i, denominator.get(j, i) +
			    point_expansion[j] * point_expansion[i] * 
			    kernel_value);
	    weight_diagram_numerator.set
	      (j, i, weight_diagram_numerator.get(j, i) +
	       point_expansion[j] * point_expansion[i] * kernel_value *
	       kernel_value);
	    
	  } // End of looping over each (j, i)-th component of the
	    // denominator matrix.
	} // End of looping over each i-th component of the numerator
	  // vector.

      } // end of looping over each reference point

      // The coefficients computed for the local fit at the given
      // query point.
      Vector beta_q;
      
      // Compute the query point expansion.
      MultiIndexUtil::ComputePointMultivariatePolynomial
	(dimension_, lpr_order_, q_col, point_expansion.ptr());

      // Now invert the denominator matrix for each query point and
      // multiply by the numerator vector.
      MatrixUtil::PseudoInverse(denominator, &denominator_inv_q);
      la::MulInit(denominator_inv_q, numerator, &beta_q);

      // Compute the dot product between the multiindex vector for the
      // query point by the beta_q.
      (*query_regression_estimates)[q] = la::Dot(beta_q, point_expansion);

      // Now we compute the magnitude of the weight diagram for each
      // query point.
      Vector pseudo_inverse_times_query_expansion, intermediate_product;
      la::MulInit(denominator_inv_q, point_expansion,
		  &pseudo_inverse_times_query_expansion);
      la::MulInit(weight_diagram_numerator,
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
    } // end of looping over each query point
  }

  void ComputeConfidenceBands_(const Matrix &queries,
			       Vector *query_regression_estimates,
			       ArrayList<DRange> *query_confidence_bands,
			       Vector *query_magnitude_weight_diagrams,
			       bool queries_equal_references) {

    // Initialize the storage for the confidene bands.
    query_confidence_bands->Init(queries.n_cols());
    
    for(index_t q = 0; q < queries.n_cols(); q++) {
      DRange &q_confidence_band = (*query_confidence_bands)[q];
      double spread;
      
      if(queries_equal_references) {
	spread = z_score_ * (*query_magnitude_weight_diagrams)[q] * 
	  sqrt(rset_variance_);
      }
      else {
	spread = z_score_ * (1 + (*query_magnitude_weight_diagrams)[q]) * 
	  sqrt(rset_variance_);
      }

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
		    Vector *query_influence_values) {

    BasicCompute_(queries, query_regression_estimates,
		  query_magnitude_weight_diagrams, query_influence_values);

    // If the reference dataset is being used for training, then
    // compute variance and degrees of freedom.
    if(query_influence_values != NULL) {
      ComputeVariance_();
    }

    ComputeConfidenceBands_(queries, query_regression_estimates,
			    query_confidence_bands,
			    query_magnitude_weight_diagrams,
			    (query_influence_values != NULL));
  }

  /** @brief Initialize the bandwidth by either fixed bandwidth
   *         parameter or a nearest neighbor based one (i.e. perform
   *         nearest neighbor and set the bandwidth equal to the k-th
   *         nearest neighbor distance).
   */
  void InitializeBandwidths_() {

    kernels_.Init(rset_.n_cols());

    if(fx_param_exists(NULL, "bandwidth")) {     
      double bandwidth = fx_param_double_req(NULL, "bandwidth");
      for(index_t i = 0; i < kernels_.size(); i++) {	
	kernels_[i].Init(bandwidth);
      }
    }
    else {
      AllkNN all_knn;
      double knn_factor = fx_param_double(module_, "knn_factor", 0.2);
      int knns = (int) (knn_factor * rset_.n_cols());
      all_knn.Init(rset_, 20, knns);
      ArrayList<index_t> resulting_neighbors;
      ArrayList<double> distances;
      
      all_knn.ComputeNeighbors(&resulting_neighbors, &distances);

      for(index_t i = 0; i < distances.size(); i += knns) {
	kernels_[i / knns].Init(distances[i]);
      }
    }
  }

 public:

  ////////// Getter/Setters //////////

  /** @brief Gets the regresion estimates of the model.
   */
  void get_regression_estimates(Vector *rset_regression_estimates_copy) {
    rset_regression_estimates_copy->Copy(rset_regression_estimates_);
  }

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

    fx_timer_start(module_, "naive_lpr_querying_time");
    ComputeMain_(queries, query_regression_estimates, query_confidence_bands, 
		 query_magnitude_weight_diagrams, query_influence_values, 
		 false);
    fx_timer_stop(module_, "naive_lpr_querying_time");
  }

  /** @brief Initialize the naive algorithm for initial usage.
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
    lpr_order_ = fx_param_int_req(NULL, "lpr_order");

    // Set the z-score
    z_score_ = fx_param_double(module_in, "z_score", 1.96);

    // Set the dimensionality.
    dimension_ = references.n_rows();

    // Copy the datasets and the reference target training values.
    rset_.Copy(references);
    rset_targets_.Copy(reference_targets.GetColumnPtr(0),
		       reference_targets.n_cols());
    
    // Get bandwidth.
    InitializeBandwidths_();

    // Compute total number of coefficients.
    total_num_coeffs_ = (int) 
      math::BinomialCoefficient(lpr_order_ + rset_.n_rows(), rset_.n_rows());
    
    // Train the model using the reference set (i.e. compute
    // confidence interval and degrees of freedom.)
    fx_timer_start(module_, "naive_lpr_training_time");
    ComputeMain_(references, &rset_regression_estimates_, 
		 &rset_confidence_bands_, &rset_magnitude_weight_diagrams_, 
		 &rset_influence_values_);
    fx_timer_stop(module_, "naive_lpr_training_time");
  }

  void PrintDebug() {

    FILE *stream = stdout;
    const char *fname = NULL;
    
    if((fname = fx_param_str(module_, "naive_lpr_output", 
			     "naive_lpr_output.txt")) != NULL) {
      stream = fopen(fname, "w+");
    }
    for(index_t r = 0; r < rset_.n_cols(); r++) {
      fprintf(stream, "%g %g %g\n", rset_confidence_bands_[r].lo,
	      rset_regression_estimates_[r], rset_confidence_bands_[r].hi);
    }
    
    if(stream != stdout) {
      fclose(stream);
    }
  }

};

#endif
