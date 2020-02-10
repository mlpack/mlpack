/**
 * @file bce_with_logits_loss.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the Binary Cross Entropy with Logits Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_BCE_WITH_LOGITS_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_BCE_WITH_LOGITS_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The binary cross-entropy with logits performance function measures the loss
 * between the input and target distributions, where the labels in target can
 * be 0 or 1. This loss combines a Sigmoid layer and the BCELoss in one single
 * class. This version is more numerically stable than using a plain Sigmoid
 * followed by a BCELoss as, by combining the operations into one layer,
 * we take advantage of the log-sum-exp trick for numerical stability.
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
class BCEWithLogitsLoss
{
 public:
  /**
   * Create the BCEWithLogitsLoss object.
   *
   * @param weight The manual rescaling factor given to loss.
   * @param reduction The boolean value, when 1, means reduction type is 'mean'
   *                  else if it is 0 then reduction type is 'sum'.
   */
  BCEWithLogitsLoss(
    const double weight = 1.0,
    const double posWeight = 1.0,
    const bool reduce = true);

  /**
   * Computes the binary cross-entropy with logits loss.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target has 1 or 0 values. If the data is
   *              multi-class then target should be one-hot encoded.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, const TargetType&& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated loss.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the weight.
  double Weight() const { return weight; }
  //! Modify the weight.
  double& Weight() { return weight; }

  //! Get the posWeight.
  double PosWeight() const { return posWeight; }
  //! Modify the posWeight.
  double& PosWeight() { return posWeight; }

  //! Get the reduce.
  bool Reduce() const { return reduce; }
  //! Modify the reduce.
  bool& Reduce() { return reduce; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! The manual rescaling factor given to the loss.
  double weight;

  //! The weight for positive examples.
  double posWeight;

  //! The boolean value that tells if reduction is mean or sum.
  bool reduce;
}; // class BCEWithLogitsLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "bce_with_logits_loss_impl.hpp"

#endif
