/**
 * @file methods/ann/loss_functions/empty_loss.hpp
 * @author Xiaohong Ji
 *
 * Definition of empty loss function. Sometimes, the user may want to calculate
 * the loss outside of the model, so we have created an empty loss that does
 * nothing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_EMPTY_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_EMPTY_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The empty loss does nothing, letting the user calculate the loss outside
 * the model.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class EmptyLossType
{
 public:
  /**
   * Create the EmptyLossType object.
   */
  EmptyLossType();

  /**
   * Computes the Empty loss function.
   *
   * @param prediction Prediction used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   */
  double Forward(const MatType& input, const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Prediction used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   * @param loss The calculated error.
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Serialize the EmptyLossType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) { }
}; // class EmptyLossType

// Default typedef for typical `arma::mat` usage.
using EmptyLoss = EmptyLossType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "empty_loss_impl.hpp"

#endif
