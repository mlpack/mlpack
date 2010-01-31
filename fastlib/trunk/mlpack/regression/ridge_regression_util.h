#ifndef RIDGE_REGRESSION_UTIL_H
#define RIDGE_REGRESSION_UTIL_H

#include "ridge_regression.h"

class RidgeRegressionUtil {

 public:

  template<typename T>
  static void CopyVectorExceptOneIndex_(const GenVector<T> &source,
					index_t exclude_index,
					GenVector<T> *destination) {
    destination->Init(source.length() - 1);
    index_t current_index = 0;

    for(index_t j = 0; j < source.length(); j++) {
      if(source[j] != exclude_index) {
	(*destination)[current_index] = source[j];
	current_index++;
      }
    }
  }

  static double SquaredCorrelationCoefficient(const Vector &observations,
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
    
    return 1.0 / 
      (1.0 - SquaredCorrelationCoefficient(observations, predictions));
  }

};

#endif
