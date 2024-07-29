/**
 * @file methods/xgboost/loss_functions/loss_functions.hpp
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

#include "log_loss.hpp"
#include "sse_loss.hpp"
#include <mlpack/core.hpp>

#ifndef MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_HPP
#define MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_HPP

namespace mlpack {

class LossFunction
{

  private:

  std::string lossType; 
  std::vector<double> losses;
  //! The L1 regularization parameter.
  const double alpha;
  //! The L2 regularization parameter.
  const double lambda;

  public:

  // Default constructor---No regularization.
  LossFunction() : alpha(0), lambda(0) { /* Nothing to do. */}

  LossFunction(std::string type) : alpha(0), lambda(0), lossType(type) { /* Nothing to do */ }

  LossFunction(const double alpha, const double lambda, std::string type) : 
    alpha(alpha), lambda(lambda), lossType(type) { /* Nothing to do */ }

  double CalculateLoss()
  {

    double loss = 0;
    if (lossType == "log_loss")
    {
      loss = EvaluateLogLoss();
    }

    else if (lossType == "sse_loss")
    {
      loss = EvaluateSSELoss();
    }

    else 
    {
      Log::Fatal << "Invalid loss Type." << std::endl;
    }

    losses.push_back(loss);
    
    return loss;
  }

  double EvaluateLogLoss()
  {

  }

  double EvaluateLogGradient()
  {
    
  }

  double EvaluateSSELoss()
  {
    
  }

  double EvaluateSSEGradient()
  {
    
  }

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

