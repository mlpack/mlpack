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
namespace ann /** Artificial Neural Network. */ {

/**
 * The empty loss does nothing, letting the user calculate the loss outside
 * the model.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class EmptyLoss
{
 public:
  /**
   * Create the EmptyLoss object.
   */
  EmptyLoss();

  /**
   * Computes the Empty loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType& input, const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType& input,
                const TargetType& target,
                OutputType& output);
}; // class EmptyLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "empty_loss_impl.hpp"

#endif
