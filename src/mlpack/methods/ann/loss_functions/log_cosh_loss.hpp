/**
 * @file methods/ann/loss_functions/log_cosh_loss.hpp
 * @author Kartik Dutt
 *
 * Definition of the Log-Hyperbolic-Cosine loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_LOG_COSH_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_LOG_COSH_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Log-Hyperbolic-Cosine loss function is often used to improve
 * variational auto encoder. This function is the log of hyperbolic
 * cosine of difference between true values and predicted values.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class LogCoshLossType
{
 public:
  /**
   * Create the Log-Hyperbolic-Cosine object with the specified
   * parameters.
   *
   * @param a A double type value for smoothening loss function. It must be a
   *          positive real number. Sharpness of loss function is directly
   *          proportional to a. It can also act as a scaling factor, hence
   *          making the loss function more sensitive to small losses around
   *          the origin. Default value = 1.0.
   * @param reduction Specifies the reduction to apply to the output. If false,
   *                  'mean' reduction is used, where sum of the output will be
   *                  divided by the number of elements in the output. If true,
   *                  'sum' reduction is used and the output will be summed. It
   *                  is set to true by default.
   */
  LogCoshLossType(const double a = 1.0, const bool reduction = true);

  /**
   * Computes the Log-Hyperbolic-Cosine loss function.
   *
   * @param prediction Predictions used for evaluating the specified loss
   *     function.
   * @param target Target data to compare with.
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

  //! Get the value of hyperparameter a.
  double A() const { return a; }
  //! Modify the value of hyperparameter a.
  double& A() { return a; }

  //! Get the reduction type, represented as boolean
  //! (false 'mean' reduction, true 'sum' reduction).
  bool Reduction() const { return reduction; }
  //! Modify the type of reduction used.
  bool& Reduction() { return reduction; }

  /**
   * Serialize the loss function.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Hyperparameter a for smoothening function curve.
  double a;

  //! Boolean value that tells if reduction is 'sum' or 'mean'.
  bool reduction;
}; // class LogCoshLossType

// Default typedef for typical `arma::mat` usage.
using LogCoshLoss = LogCoshLossType<arma::mat>;

} // namespace mlpack

// include implementation
#include "log_cosh_loss_impl.hpp"

#endif
