#ifndef RIDGE_REGRESSCLIN_UTIL_H
#define RIDGE_REGRESSCLIN_UTIL_H

#include "ridge_regression.h"

namespace mlpack {
namespace regression {

class RidgeRegressionUtil {

 public:

  template<typename T>
  static void CopyVectorExceptOneIndex_(const arma::Col<T> &source,
					size_t exclude_index,
					arma::Col<T> *destination) {
    destination->zeros(source.n_elem - 1);
    size_t current_index = 0;

    for(size_t j = 0; j < source.n_elem; j++) {
      if(source[j] != exclude_index) {
	(*destination)[current_index] = source[j];
	current_index++;
      }
    }
  }

  static double SquaredCorrelationCoefficient(const arma::vec &observations,
					      const arma::vec &predictions) {

    // Compute the average of the observed values.
    double avg_observed_value = 0;

    for(size_t i = 0; i < observations.n_elem; i++) {
      avg_observed_value += observations[i];
    }
    avg_observed_value /= ((double) observations.n_elem);

    // Compute something proportional to the variance of the observed
    // values, and the sum of squared residuals of the predictions
    // against the observations.
    double variance = 0;
    double residual = 0;
    for(size_t i = 0; i < observations.n_elem; i++) {
      variance += pow(observations[i] - avg_observed_value, 2);
      residual += pow(observations[i] - predictions[i], 2);
    }
    return (variance - residual) / variance;
  }

  static double VarianceInflationFactor(const arma::vec &observations,
					const arma::vec &predictions) {

    return 1.0 /
      (1.0 - SquaredCorrelationCoefficient(observations, predictions));
  }

};

}; // namespace regression
}; // namespace mlpack

#endif
