/**
 * @file methods/decision_tree/gain_functions/log_gain.hpp
 * @author Abhimanyu Dayal
 *
 * The logistic gain class, which is a gain function for xgboost.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_LOG_LOSS_HPP
#define MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_LOG_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
/**
 * Logistic loss, also known as log loss or cross-entropy loss, is a loss 
 * function used in logistic regression to measure the difference between 
 * the predicted probability of an event and the actual outcome.
 * 
 * Log Loss = - (1 / N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
 */
class LogLoss
{
  public:
  // Default constructor---No regularization.
  LogLoss() : alpha(0), lambda(0) { /* Nothing to do. */}

  LogLoss(const double alpha, const double lambda):
      alpha(alpha), lambda(lambda)
  {
    // Nothing to do.
  }

  /**
   * Returns the initial prediction for gradient boosting.
   */
  template<typename VecType>
  typename VecType::elem_type InitialPrediction(const VecType& values)
  {
    // Sanity check for empty vector.
    if (values.n_elem == 0)
      return 0;

    // Return the log-odds of the mean of the values.
    double mean = accu(values) / (typename VecType::elem_type) values.n_elem;
    return std::log(mean / (1 - mean));
  }


  private:
  //! The L1 regularization parameter.
  const double alpha;
  //! The L2 regularization parameter.
  const double lambda;
  //! First order gradients.
  arma::vec gradients;
  //! Second order gradients (hessians).
  arma::vec hessians;

  //! Applies the L1 regularization.
  double ApplyL1(const double sumGradients)
  {
    if (sumGradients > alpha)
    {
      return sumGradients - alpha;
    }
    else if (sumGradients < - alpha)
    {
      return sumGradients + alpha;
    }

    return 0;
  }

};
} // namespace mlpack

#endif
