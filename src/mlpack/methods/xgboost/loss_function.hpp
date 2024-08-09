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
 * Unified Loss Function containing the different loss functions that can be run 
 * based on the user's choice. Contains the following: 
 * 
 * Log Loss:
 * Logistic loss, also known as log loss or cross-entropy loss, is a loss 
 * function used in logistic regression to measure the difference between 
 * the predicted probability of an event and the actual outcome.
 * 
 * Log Loss = - (1 / N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
 * 
 * SSE Loss:
 * The SSE (Sum of Squared Errors) loss is a loss function to measure the
 * quality of prediction of response values present in the node of each
 * xgboost tree. It is also a good measure to compare the spread of two
 * distributions. We will try to minimize this value while training.
 *
 * Loss = 1 / 2 * (Observed - Predicted)^2
 */
class LossFunction
{

  public:

  // Default constructor---No regularization.
  LossFunction() : alpha(0), lambda(0) { /* Nothing to do. */}

  LossFunction(std::string type) : alpha(0), lambda(0), lossType(type) { /* Nothing to do */ }

  LossFunction(const double alpha, const double lambda, std::string type) : 
    alpha(alpha), lambda(lambda), lossType(type) { /* Nothing to do */ }

  double EvaluateLoss(const arma::mat& input)
  {
    // Initiate loss variable to store calculated loss value.
    double loss = 0;

    // Determine which loss function to determine
    if (lossType == "log_loss")
      loss = EvaluateLogLoss(input);

    else if (lossType == "sse_loss")
      loss = EvaluateSSELoss(input);

    // Throw an error if loss type isn't specified. 
    else 
      Log::Fatal << "Invalid loss Type." << std::endl;

    losses.push_back(loss);
    
    return loss;
  }

  double EvaluateLogLoss(const arma::mat& input)
  {

  }

  double EvaluateLogGradient()
  {
    
  }

  /**
   * Calculate the SSE Loss. 
   * 
   * Loss = 1 / 2 * (Observed - Predicted)^2
   * 
   * Also has L1 and L2 regularisations using the alpha and lambda values. 
   *
   * @param input This is a 2D matrix. The first row stores the true observed
   *    values and the second row stores the prediction at the current step
   *    of boosting.
   */
  double EvaluateSSELoss(const arma::mat& input)
  {
    // Calculate gradients and hessians.
    gradients = (input.row(1) - input.row(0)).t();
    hessians = arma::vec(input.n_cols, arma::fill::ones);

    // Apply L1 regularization to the sum of gradients.
    double sumGradients = accu(gradients);
    double regGradients = ApplyL1(sumGradients);

    // Calculate the loss using the regularized gradients and hessians.
    double loss = std::pow(regGradients, 2) / (accu(hessians) + lambda);

    return loss;
  }

  double EvaluateSSEGradient()
  {
    
  }


  private:

  //! Specifies the loss type. Can take values: `log_loss` and `sse_loss`
  std::string lossType; 

  //! Stores losses calculated at all iterations.
  std::vector<double> losses;

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

