/**
 * @file smooth_l1_loss.hpp
 * @author Saksham Rastogi
 *
 * Definition of the Smooth L1 Loss Function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SMOOTH_L1_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SMOOTH_L1_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
    namespace ann /** Artificial Neural Network. */ {

/**
 * The Smooth L1 Loss Function computes the Smooth L1 distance between 
 * each element in the input x and target y
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class SmoothL1Loss {
 public:
  /**
   * Create the Smooth L1 Loss Object
   */
  SmoothL1Loss(
      const bool takeMean = true,
      const double sigma = 1.0
    );

  /**
   * Computes the mean squared error function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, const TargetType&& target);
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);
  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the value of takeMean.
  bool TakeMean() const { return takeMean; }
  //! Modify the value of takeMean.
  bool& TakeMean() { return takeMean; }

  //! Get the value of takeMean.
  double Sigma() const { return sigma; }
  //! Modify the value of takeMean.
  double& Sigma() { return sigma; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Boolean variable for taking mean or not.
  bool takeMean;

  //! Parameter to decide whether to make MSE or MAE.
  double sigma;
}; // class SmoothL1Loss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "smooth_l1_loss_impl.hpp"

#endif