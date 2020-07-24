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
namespace ann /** Artificial Neural Network. */ {

/**
 * The Huber loss is a loss function used in robust regression,
 * that is less sensitive to outliers in data than the squared error loss.
 * This function is quadratic for small values of \f$ y - f(x) \f$,
 * and linear for large values, with equal values and slopes of the different
 * sections at the two points where \f$ |y - f(x)| = delta \f$.
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
class HuberLoss
{
 public:
  /**
   * Create the HuberLoss object.
   *
   * @param delta The threshold value upto which squared error is followed and
   *              after which absolute error is considered.
   * @param mean If true then mean loss is computed otherwise sum.
   */
  HuberLoss(const double delta = 1.0, const bool mean = true);

  /**
   * Computes the Huber Loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
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

  //! Get the value of delta.
  double Delta() const { return delta; }
  //! Set the value of delta.
  double& Delta() { return delta; }

  //! Get the value of reduction type.
  bool Mean() const { return mean; }
  //! Set the value of reduction type.
  bool& Mean() { return mean; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Hyperparameter `delta` defines the point upto which MSE is considered.
  double delta;

  //! Reduction type. If true, performs mean of loss else sum.
  bool mean;
}; // class HuberLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "huber_loss_impl.hpp"

#endif
