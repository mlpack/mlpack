/**
 * @file methods/ann/layer/base_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BaseLayer class, which attaches various functions to the
 * embedding layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>
#include <mlpack/methods/ann/activation_functions/swish_function.hpp>
#include <mlpack/methods/ann/activation_functions/mish_function.hpp>
#include <mlpack/methods/ann/activation_functions/lisht_function.hpp>
#include <mlpack/methods/ann/activation_functions/gelu_function.hpp>
#include <mlpack/methods/ann/activation_functions/elliot_function.hpp>
#include <mlpack/methods/ann/activation_functions/elish_function.hpp>
#include <mlpack/methods/ann/activation_functions/gaussian_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the base layer. The base layer works as a metaclass which
 * attaches various functions to the embedding layer.
 *
 * A few convenience typedefs are given:
 *
 *  - SigmoidLayer
 *  - IdentityLayer
 *  - ReLULayer
 *  - TanHLayer
 *  - SoftplusLayer
 *  - HardSigmoidLayer
 *  - SwishLayer
 *  - MishLayer
 *  - LiSHTLayer
 *  - GELULayer
 *  - ELiSHLayer
 *  - ElliotLayer
 *  - GaussianLayer
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    class ActivationFunction = LogisticFunction,
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class BaseLayer : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the BaseLayer object.
   */
  BaseLayer()
  {
    // Nothing to do here.
  }

  //! Clone the BaseLayer object. This handles polymorphism correctly.
  BaseLayer* Clone() const { return new BaseLayer(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output)
  {
    ActivationFunction::Fn(input, output);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g)
  {
    OutputType derivative;
    ActivationFunction::Deriv(input, derivative);
    g = gy % derivative;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(cereal::base_class<Layer<InputType, OutputType>>(this));
    // Nothing to serialize.
  }
}; // class BaseLayer

// Convenience typedefs.

/**
 * Standard Sigmoid-Layer using the logistic activation function.
 */
typedef BaseLayer<LogisticFunction, arma::mat, arma::mat> Sigmoid;
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using SigmoidLayer = BaseLayer<LogisticFunction, InputType, OutputType>;

/**
 * Standard Identity-Layer using the identity activation function.
 */
typedef BaseLayer<IdentityFunction, arma::mat, arma::mat> Identity;
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using IdentityLayer = BaseLayer<IdentityFunction, InputType, OutputType>;

/**
 * Standard rectified linear unit non-linearity layer.
 */
typedef BaseLayer<RectifierFunction, arma::mat, arma::mat> ReLU;
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using ReLULayer = BaseLayer<RectifierFunction, InputType, OutputType>;

/**
 * Standard hyperbolic tangent layer.
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using TanHLayer = BaseLayer<TanhFunction, InputType, OutputType>;

/**
 * Standard Softplus-Layer using the Softplus activation function.
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using SoftPlusLayer = BaseLayer<SoftplusFunction, InputType, OutputType>;

/**
 * Standard HardSigmoid-Layer using the HardSigmoid activation function.
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using HardSigmoidLayer = BaseLayer<HardSigmoidFunction, InputType, OutputType>;

/**
 * Standard Swish-Layer using the Swish activation function.
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using SwishFunctionLayer = BaseLayer<SwishFunction, InputType, OutputType>;

/**
 * Standard Mish-Layer using the Mish activation function.
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using MishFunctionLayer = BaseLayer<MishFunction, InputType, OutputType>;

/**
 * Standard LiSHT-Layer using the LiSHT activation function.
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using LiSHTFunctionLayer = BaseLayer<LiSHTFunction, InputType, OutputType>;

/**
 * Standard GELU-Layer using the GELU activation function.
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using GELUFunctionLayer = BaseLayer<GELUFunction, InputType, OutputType>;

/**
 * Standard Elliot-Layer using the Elliot activation function.
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using ElliotFunctionLayer = BaseLayer<ElliotFunction, InputType, OutputType>;

/**
 * Standard ELiSH-Layer using the ELiSH activation function.
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using ElishFunctionLayer = BaseLayer<ElishFunction, InputType, OutputType>;

/**
 * Standard Gaussian-Layer using the Gaussian activation function.
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
using GaussianFunctionLayer = BaseLayer<GaussianFunction, InputType,
    OutputType>;

} // namespace ann
} // namespace mlpack

#endif
