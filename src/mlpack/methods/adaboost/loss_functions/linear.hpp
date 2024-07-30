/**
 * @file methods/adaboost/loss_functions/linear.hpp
 * @author Dinesh Kumar
 * 
 * Linear loss class, which is a loss function for adaboost regressor.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ADABOOST_LOSS_FUNCTIONS_LINEAR_HPP
#define MLPACK_METHODS_ADABOOST_LOSS_FUNCTIONS_LINEAR_HPP

#include "mlpack.hpp"

namespace mlpack {
 
/**
* The linear loss is a loss function to measure the Beta value of the 
* predictor and update the weights for the next machine in the ensemble.
*
* Loss = | error_values | / max(error_values)
*/
class LinearLoss
{
  public:
  template <typename VecType>
  static VecType Calculate (const VecType& error_vec)
  {
    typename VecType::elem_type max_error = arma::max(error_vec);
    // To avoid dividing by zero
    if(max_error == 0)
      max_error = 1;
    VecType loss = arma::abs(error_vec) / max_error;
    return loss;
  }
};
}

#endif