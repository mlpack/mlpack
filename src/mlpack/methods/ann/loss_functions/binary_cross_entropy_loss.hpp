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

/**
 * The binary-cross-entropy performance function measures the Binary Cross
 * Entropy between the target and the output.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class BCELossType
{
 public:
  /**
   * Create the BinaryCrossEntropyLoss object.
   *
   * @param eps The minimum value used for computing logarithms
   *            and denominators in a numerically stable way.
   *
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  BCELossType(const double eps = 1e-10, const bool reduction = true);

  /**
   * Computes the cross-entropy function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   */
  typename MatType::elem_type Forward(const MatType& prediction,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target The target vector.
   * @param loss The calculated error.
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Get the epsilon.
  double Eps() const { return eps; }
  //! Modify the epsilon.
  double& Eps() { return eps; }

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! The minimum value used for computing logarithms and denominators
  double eps;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class BCELossType

// Default typedef for typical `arma::mat` usage.
using BCELoss = BCELossType<arma::mat>;

/**
 * Alias of BCELossType.
 */
using CrossEntropyError = BCELossType<arma::mat>;

template<typename MatType = arma::mat>
using CrossEntropyErrorType = BCELossType<MatType>;

} // namespace mlpack

// Include implementation.
#include "binary_cross_entropy_loss_impl.hpp"

#endif
