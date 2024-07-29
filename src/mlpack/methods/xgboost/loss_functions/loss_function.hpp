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

  LossFunction() {/*Nothing to do*/}

  LossFunction(std::string type) : lossType(type) {/*Nothing to do*/}

  void CalculateLoss()
  {
    if (lossType == "log_loss")
    {
      
    }
  }

  double EvaluateLogLoss()
  {

  }



  double EvaluateSSELoss()
  {
    
  }
};

} // namespace mlpack

#endif

