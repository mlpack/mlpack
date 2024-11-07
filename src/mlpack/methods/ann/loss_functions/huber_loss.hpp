/**
 * @file methods/ann/loss_functions/huber_loss.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the Huber loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Huber loss is a loss function used in robust regression,
 * that is less sensitive to outliers in data than the squared error loss.
 * This function is quadratic for small values of \f$ y - f(x) \f$,
 * and linear for large values, with equal values and slopes of the different
 * sections at the two points where \f$ |y - f(x)| = delta \f$.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class HuberLossType
{
 public:
  /**
   * Create the HuberLossType object.
   *
   * @param delta The threshold value upto which squared error is followed and
   *              after which absolute error is considered.
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  HuberLossType(const double delta = 1.0, const bool reduction = true);

  /**
   * Computes the Huber Loss function.
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

  //! Get the value of delta.
  double Delta() const { return delta; }
  //! Set the value of delta.
  double& Delta() { return delta; }

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
  //! Hyperparameter `delta` defines the point upto which MSE is considered.
  double delta;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class HuberLossType

// Default typedef for typical `arma::mat` usage.
using HuberLoss = HuberLossType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "huber_loss_impl.hpp"

#endif
