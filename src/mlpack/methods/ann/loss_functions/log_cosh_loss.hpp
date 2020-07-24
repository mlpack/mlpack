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
namespace ann /** Artificial Neural Network. */ {

/**
 * The Log-Hyperbolic-Cosine loss function is often used to improve
 * variational auto encoder. This function is the log of hyperbolic
 * cosine of difference between true values and predicted values.
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
class LogCoshLoss
{
 public:
  /**
   * Create the Log-Hyperbolic-Cosine object with the specified
   * parameters.
   *
   * @param a A double type value for smoothening loss function.
   *          It must be positive a real number, Sharpness of loss
   *          function is directly proportional to a. It can also
   *          act as a scaling factor hence making the loss
   *          function more sensitive to small losses around the
   *          origin. Default value = 1.0.
   */
  LogCoshLoss(const double a = 1.0);

  /**
   * Computes the Log-Hyperbolic-Cosine loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target Target data to compare with.
   */
  template<typename InputType, typename TargetType>
  typename InputType::elem_type Forward(const InputType& input,
                                        const TargetType& target);

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

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the value of hyperparameter a.
  double A() const { return a; }
  //! Modify the value of hyperparameter a.
  double& A() { return a; }

  /**
   * Serialize the loss function.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Hyperparameter a for smoothening function curve.
  double a;
}; // class LogCoshLoss

} // namespace ann
} // namespace mlpack

// include implementation
#include "log_cosh_loss_impl.hpp"

#endif
