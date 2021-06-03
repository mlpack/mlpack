/**
 * @file methods/ann/loss_functions/cross_entropy_error.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the binary-cross-entropy performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_ERROR_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The binary-cross-entropy performance function measures the
 * Binary Cross Entropy between the target and the output.
 * This function calculates the Binary Cross Entropy between input and target,
 * and expects the target to be one-hot encoded and the input matrix to only
 * have values between 0 and 1, both inclusive.
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
class BCELoss
{
 public:
  /**
   * Create the BinaryCrossEntropyLoss object.
   *
   * @param eps The minimum value used for computing logarithms
   *            and denominators in a numerically stable way.
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If
   *                  true, 'sum' reduction is used and the output will be
   *                  summed. It is set to true by default.
   */
  BCELoss(const double eps = 1e-10, const bool reduction = true);

  /**
   * Computes the cross-entropy function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   */
  template<typename PredictionType, typename TargetType>
  typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                             const TargetType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   * @param loss The calculated error.
   */
  template<typename PredictionType, typename TargetType, typename LossType>
  void Backward(const PredictionType& prediction,
                const TargetType& target,
                LossType& loss);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the epsilon.
  double Eps() const { return eps; }
  //! Modify the epsilon.
  double& Eps() { return eps; }

  //! Get the type of reduction used.
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! The minimum value used for computing logarithms and denominators
  double eps;

  //! The boolean value that tells if reduction is sum or mean.
  bool reduction;
}; // class BCELoss

/**
 * Adding alias of BCELoss.
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
using CrossEntropyError = BCELoss<
    InputDataType, OutputDataType>;

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "binary_cross_entropy_loss_impl.hpp"

#endif
