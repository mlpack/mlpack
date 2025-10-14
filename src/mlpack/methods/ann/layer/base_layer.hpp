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
#include <mlpack/methods/ann/activation_functions/gelu_exact_function.hpp>
#include <mlpack/methods/ann/activation_functions/elliot_function.hpp>
#include <mlpack/methods/ann/activation_functions/elish_function.hpp>
#include <mlpack/methods/ann/activation_functions/gaussian_function.hpp>
#include <mlpack/methods/ann/activation_functions/hard_swish_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_exponential_function.hpp>
#include <mlpack/methods/ann/activation_functions/silu_function.hpp>
#include <mlpack/methods/ann/activation_functions/hyper_sinh_function.hpp>
#include <mlpack/methods/ann/activation_functions/bipolar_sigmoid_function.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the base layer. The base layer works as a metaclass which
 * attaches various functions to the embedding layer.
 *
 * A few convenience typedefs are given:
 *
 *  - Sigmoid
 *  - ReLU
 *  - TanH
 *  - Softplus
 *  - HardSigmoid
 *  - Swish
 *  - Mish
 *  - LiSHT
 *  - GELU
 *  - GELUExact
 *  - ELiSH
 *  - Elliot
 *  - Gaussian
 *  - HardSwish
 *  - TanhExp
 *  - SILU
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat,
    bool DerivRequiresOutput = true
>
class BaseLayer : public Layer<MatType>
{
 public:
  // Convenience typedef to access the element type of the weights and data.
  using ElemType = typename MatType::elem_type;

  /**
   * Create the BaseLayer object.
   */
  BaseLayer() : Layer<MatType>()
  {
    // Nothing to do here.
  }

  // Virtual destructor.
  virtual ~BaseLayer() { }

  // No copy constructor or operators needed here, since the class has no
  // members.

  //! Clone the BaseLayer object. This handles polymorphism correctly.
  virtual BaseLayer* Clone() const { return new BaseLayer(*this); }

  /**
   * Forward pass: apply the activation to the inputs.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output)
  {
    ActivationFunction::Fn(input, output);
  }

  /**
   * Backward pass: compute the function f(x) by propagating x backwards through
   * f, using the results from the forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& output,
                const MatType& gy,
                MatType& g)
  {
    MatType derivative;
    ActivationFunction::Deriv(input, output, derivative);
    g = gy % derivative;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(cereal::base_class<Layer<MatType>>(this));
    // Nothing to serialize.
  }
}; // class BaseLayer

// Convenience typedefs.

/**
 * Standard Sigmoid-Layer using the logistic activation function.
 */
template<typename MatType = arma::mat>
using Sigmoid = BaseLayer<LogisticFunction, MatType>;

/**
 * Standard rectified linear unit non-linearity layer.
 */
template<typename MatType = arma::mat>
using ReLU = BaseLayer<RectifierFunction, MatType>;

/**
 * Standard hyperbolic tangent layer.
 */
template<typename MatType = arma::mat>
using TanH = BaseLayer<TanhFunction, MatType>;

/**
 * Standard Softplus-Layer using the Softplus activation function.
 */
template<typename MatType = arma::mat>
using SoftPlus = BaseLayer<SoftplusFunction, MatType>;

/**
 * Standard HardSigmoid-Layer using the HardSigmoid activation function.
 */
template<typename MatType = arma::mat>
using HardSigmoid = BaseLayer<HardSigmoidFunction, MatType>;

/**
 * Standard Swish-Layer using the Swish activation function.
 */
template<typename MatType = arma::mat>
using Swish = BaseLayer<SwishFunction, MatType>;

/**
 * Standard Mish-Layer using the Mish activation function.
 */
template<typename MatType = arma::mat>
using Mish = BaseLayer<MishFunction, MatType>;

/**
 * Standard LiSHT-Layer using the LiSHT activation function.
 */
template<typename MatType = arma::mat>
using LiSHT = BaseLayer<LiSHTFunction, MatType>;

/**
 * Standard GELU-Layer using the GELU activation function.
 */
template<typename MatType = arma::mat>
using GELU = BaseLayer<GELUFunction, MatType>;

/**
 * Standard GELUExact-Layer using the GELUExact activation function.
 */
template<typename MatType = arma::mat>
using GELUExact = BaseLayer<GELUExactFunction, MatType>;

/**
 * Standard Elliot-Layer using the Elliot activation function.
 */
template<typename MatType = arma::mat>
using Elliot = BaseLayer<ElliotFunction, MatType>;

/**
 * Standard ELiSH-Layer using the ELiSH activation function.
 */
template<typename MatType = arma::mat>
using Elish = BaseLayer<ElishFunction, MatType>;

/**
 * Standard Gaussian-Layer using the Gaussian activation function.
 */
template<typename MatType = arma::mat>
using Gaussian = BaseLayer<GaussianFunction, MatType>;

/**
 * Standard HardSwish-Layer using the HardSwish activation function.
 */
template <typename MatType = arma::mat>
using HardSwish = BaseLayer<HardSwishFunction, MatType>;

/**
 * Standard TanhExp-Layer using the TanhExp activation function.
 */
template<typename MatType = arma::mat>
using TanhExp = BaseLayer<TanhExpFunction, MatType>;

/**
 * Standard SILU-Layer using the SILU activation function.
 */
template<typename MatType = arma::mat>
using SILU = BaseLayer<SILUFunction, MatType>;

/**
 * Standard Hyper Sinh layer.
 */
template<typename MatType = arma::mat>
using HyperSinh = BaseLayer<HyperSinhFunction, MatType>;

/**
 * Standard Bipolar Sigmoid layer.
 */
template<typename MatType = arma::mat>
using BipolarSigmoid = BaseLayer<BipolarSigmoidFunction, MatType>;

} // namespace mlpack

#endif
