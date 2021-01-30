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

  //! Get the output parameter.
  OutputType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */)
  {
    /* Nothing to do here */
  }

 private:
  //! Locally-stored delta object.
  OutputType delta;

  //! Locally-stored output parameter object.
  OutputType outputParameter;
}; // class BaseLayer

// Convenience typedefs.

/**
 * Standard Sigmoid-Layer using the logistic activation function.
 */
// template <
//     class ActivationFunction = LogisticFunction,
//     typename InputType = arma::mat,
//     typename OutputType = arma::mat
// >
// using SigmoidLayer = BaseLayer<ActivationFunction, InputType, OutputType>;
typedef BaseLayer<LogisticFunction, arma::mat, arma::mat> Sigmoid;
typedef BaseLayer<LogisticFunction, arma::mat, arma::mat> SigmoidLayer;

/**
 * Standard Identity-Layer using the identity activation function.
 */
// template <
//     class ActivationFunction = IdentityFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using IdentityLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<IdentityFunction, arma::mat, arma::mat> IdentityLayer;

/**
 * Standard rectified linear unit non-linearity layer.
 */
// template <
//     class ActivationFunction = RectifierFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using ReLULayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<RectifierFunction, arma::mat, arma::mat> ReLULayer;

/**
 * Standard hyperbolic tangent layer.
 */
// template <
//     class ActivationFunction = TanhFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using TanHLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<TanhFunction, arma::mat, arma::mat> TanHLayer;

/**
 * Standard Softplus-Layer using the Softplus activation function.
 */
// template <
//     class ActivationFunction = SoftplusFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using SoftPlusLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<SoftplusFunction, arma::mat, arma::mat> SoftPlusLayer;

/**
 * Standard HardSigmoid-Layer using the HardSigmoid activation function.
 */
// template <
//     class ActivationFunction = HardSigmoidFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using HardSigmoidLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<HardSigmoidFunction, arma::mat, arma::mat> HardSigmoidLayer;

/**
 * Standard Swish-Layer using the Swish activation function.
 */
// template <
//     class ActivationFunction = SwishFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using SwishFunctionLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<SwishFunction, arma::mat, arma::mat> SwishFunctionLayer;

/**
 * Standard Mish-Layer using the Mish activation function.
 */
// template <
//     class ActivationFunction = MishFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using MishFunctionLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<MishFunction, arma::mat, arma::mat> MishFunctionLayer;

/**
 * Standard LiSHT-Layer using the LiSHT activation function.
 */
// template <
//     class ActivationFunction = LiSHTFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using LiSHTFunctionLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<LiSHTFunction, arma::mat, arma::mat> LiSHTFunctionLayer;

/**
 * Standard GELU-Layer using the GELU activation function.
 */
// template <
//     class ActivationFunction = GELUFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using GELUFunctionLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<GELUFunction, arma::mat, arma::mat> GELUFunctionLayer;

/**
 * Standard Elliot-Layer using the Elliot activation function.
 */
// template <
//     class ActivationFunction = ElliotFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using ElliotFunctionLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<ElliotFunction, arma::mat, arma::mat> ElliotFunctionLayer;

/**
 * Standard ELiSH-Layer using the ELiSH activation function.
 */
// template <
//     class ActivationFunction = ElishFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using ElishFunctionLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<ElishFunction, arma::mat, arma::mat> ElishFunctionLayer;

/**
 * Standard Gaussian-Layer using the Gaussian activation function.
 */
// template <
//     class ActivationFunction = GaussianFunction,
//     typename InputDataType = arma::mat,
//     typename OutputDataType = arma::mat
// >
// using GaussianFunctionLayer = BaseLayer<
//     ActivationFunction, InputDataType, OutputDataType>;
typedef BaseLayer<GaussianFunction, arma::mat, arma::mat> GaussianFunctionLayer;

} // namespace ann
} // namespace mlpack

#endif
