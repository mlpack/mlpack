#ifndef RIDGE_REGRESSION_UTIL_H
#define RIDGE_REGRESSION_UTIL_H

#include "ridge_regression.h"

class RidgeRegressionUtil {

 public:

  static double CorrelationCoefficient(const Vector &observations,
				       const Vector &predictions) {
    
    // Compute the average of the observed values.
    double avg_observed_value = 0;
    
    for(index_t i = 0; i < observations.length(); i++) {
      avg_observed_value += observations[i];
    }
    avg_observed_value /= ((double) observations.length());

    // Compute something proportional to the variance of the observed
    // values, and the sum of squared residuals of the predictions
    // against the observations.
    double variance = 0;
    double residual = 0;
    for(index_t i = 0; i < observations.length(); i++) {
      variance += math::Sqr(observations[i] - avg_observed_value);
      residual += math::Sqr(observations[i] - predictions[i]);
    }
    return (variance - residual) / variance;
  }

  static double VarianceInflationFactor(const Vector &observations,
					const Vector &predictions) {
    
    return 1.0 / (1.0 - CorrelationCoefficient(observations, predictions));
  }

  
  /** @brief Performs the feature selection using the variance
   *         inflation factor.
   *
   *  @param input_features The input features.
   *  @param output_feature The output features excluding the features that
   *         were pruned.
   */
  static void FeatureSelection
  (fx_module *module, const Matrix &input_featuers, 
   const GenVector<index_t> &predictor_indices, index_t prediction_index,
   double variance_inflation_factor_threshold = 8.0) {
    
    bool done_flag = false;

    do {

      // For each of the features in the current list, regress the
      // i-th feature versus the rest of the features and compute its
      // variance inflation factor.
      for(index_t i = 0; i < pruned_features.n_cols(); i++) {
	
	RidgeRegression ridge_regression;
	
      }
    } while(!done_flag);
  }

};

#endif
