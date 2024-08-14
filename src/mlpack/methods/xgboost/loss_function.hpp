/**
 * @file methods/xgboost/loss_functions.hpp
 * @author Abhimanyu Dayal
 *
 * Interface for customisable loss functions. Based on the user's choice, a 
 * specific loss function is executed.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#ifndef MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_HPP
#define MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_HPP

namespace mlpack {

/**
 * Logistic loss, also known as log loss or cross-entropy loss, is a loss 
 * function used in logistic regression to measure the difference between 
 * the predicted probability of an event and the actual outcome.
 * 
 * Log Loss = - (1 / N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
 * 
 */
double EvaluateLogLoss(const arma::Row<size_t>& labels, const arma::mat& probabilities)
{
  double n = (double) probabilities.n_cols;
  size_t numClasses = probabilities.n_rows;

  double loss = 0;

  for (size_t i = 0; i < n; ++i)
    loss -= std::log(probabilities(labels(i)));

  loss /= n; 

  return loss;
}

} // namespace mlpack

#endif

