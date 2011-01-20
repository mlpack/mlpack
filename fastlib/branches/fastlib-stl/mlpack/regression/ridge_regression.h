#ifndef RIDGE_REGRESSION_H_
#define RIDGE_REGRESSION_H_

#include "fastlib/fastlib.h"
#include "mlpack/quicsvd/quicsvd.h"
#include "ridge_regression_util.h"

class RidgeRegression {
 public:

  RidgeRegression() {
  }

  void Init(fx_module *module, const arma::mat &predictors, 
	    const arma::mat &predictions, 
	    bool use_normal_equation_method = true);

  void Init(fx_module *module, const arma::mat &input_data, index_t selector,
	    bool use_normal_equation_method = true);

  /** @brief From a column-oriented dataset, initialize the design
   *         matrix using the row features whose indices belong to the
   *         given index set, and the predicted values from the given
   *         prediction index.
   *
   *  @param module The module from which the parameters are passed
   *  from.
   *
   *  @param input_data The entire dataset containing the features and
   *  the prediction values.
   *
   *  @param predictor_indices The row indices of the input data that
   *  should be part of the design matrix.
   *
   *  @param prediction_index The row index of the input data that
   *  should be used as the predictions (training target).
   */
  void Init(fx_module *module, 
            const arma::mat &input_data, 
            const arma::Col<index_t> &predictor_indices,
            index_t &prediction_index, bool use_normal_equation_method = true);

  void Init(fx_module *module, 
            const arma::mat &input_data, 
            const arma::Col<index_t> &predictor_indices,
            const arma::mat &prediction, bool use_normal_equation_method = true);

  void ReInitTargetValues(const arma::mat &input_data, 
			  index_t target_value_index);

  void ReInitTargetValues(const arma::mat &target_values_in);

  void Destruct();

  void SVDRegress(double lambda,
		  const arma::Col<index_t> *predictor_indices = NULL);

  void QRRegress(double lambda,
		 const arma::Col<index_t> *predictor_indices = NULL);

  void CrossValidatedRegression(double lambda_min, double lambda_max,
				index_t num);

  void FeatureSelectedRegression
  (const arma::Col<index_t> &predictor_indices, 
   const arma::Col<index_t> &prune_predictor_indices, 
   const arma::mat &original_target_training_values,
   arma::Col<index_t> *output_predictor_indices);

  double ComputeSquareError();

  /** @brief Predict on another dataset given the trained linear
   *         model.
   */
  void Predict(const arma::mat &dataset, arma::vec *new_predictions);

  void Predict(const arma::mat &dataset, 
	       const arma::Col<index_t> &predictor_indices,
	       arma::vec *new_predictions);

  void factors(arma::mat *factors);

 private:

  fx_module *module_;

  /** @brief The design matrix.
   */
  arma::mat const* predictors_;

  /** @brief The training target values for each instance. This is
   *         generalizable to multi-target case by using a matrix.
   */
  arma::mat predictions_;

  /** @brief The covariance matrix: roughly A A^T, modulo the
   *         mandatory 1 vector added to the features for the
   *         intercept.
   */
  arma::mat covariance_;

  /** @brief The trained linear regression model output.
   */
  arma::mat factors_;

  void ComputeLinearModel_(double lambda_sq, const arma::vec &singular_values, 
			   const arma::mat &u, const arma::mat &v_t,
			   int num_features);
  
  void BuildDesignMatrixFromIndexSet_
  (const arma::mat &input_data, const arma::mat& predictions,
   const arma::Col<index_t> *predictor_indices);
  
  void BuildCovariance_(const arma::mat &input_data, 
			const arma::Col<index_t> *predictor_indices,
			const arma::mat& predictions_in);

  void ExtractCovarianceSubset_
  (const arma::mat &precomputed_covariance,
   const arma::Col<index_t> *loo_current_predictor_indices,
   arma::mat *precomputed_covariance_subset);

  void ExtractDesignMatrixSubset_
  (const arma::Col<index_t> *loo_current_predictor_indices,
   arma::mat *extracted_design_matrix_subset);

  void ExtractSubspace_(arma::mat &u, arma::vec &singular_values, arma::mat &v_t,
			const arma::Col<index_t> *predictor_indices);
};

#include "ridge_regression_impl.h"
#endif
