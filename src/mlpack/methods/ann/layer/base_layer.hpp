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
  BaseLayer* Clone() const { return new BaseLayer(*this); }

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
using Sigmoid = BaseLayer<LogisticFunction, arma::mat>;

template<typename MatType = arma::mat>
using SigmoidType = BaseLayer<LogisticFunction, MatType>;

/**
 * Standard rectified linear unit non-linearity layer.
 */
using ReLU = BaseLayer<RectifierFunction, arma::mat>;

template<typename MatType = arma::mat>
using ReLUType = BaseLayer<RectifierFunction, MatType>;

/**
 * Standard hyperbolic tangent layer.
 */
using TanH = BaseLayer<TanhFunction, arma::mat>;

template<typename MatType = arma::mat>
using TanHType = BaseLayer<TanhFunction, MatType>;

/**
 * Standard Softplus-Layer using the Softplus activation function.
 */
using SoftPlus = BaseLayer<SoftplusFunction, arma::mat>;

template<typename MatType = arma::mat>
using SoftPlusType = BaseLayer<SoftplusFunction, MatType>;

/**
 * Standard HardSigmoid-Layer using the HardSigmoid activation function.
 */
using HardSigmoid = BaseLayer<HardSigmoidFunction, arma::mat>;

template<typename MatType = arma::mat>
using HardSigmoidType = BaseLayer<HardSigmoidFunction, MatType>;

/**
 * Standard Swish-Layer using the Swish activation function.
 */
using Swish = BaseLayer<SwishFunction, arma::mat>;

template<typename MatType = arma::mat>
using SwishType = BaseLayer<SwishFunction, MatType>;

/**
 * Standard Mish-Layer using the Mish activation function.
 */
using Mish = BaseLayer<MishFunction, arma::mat>;

template<typename MatType = arma::mat>
using MishType = BaseLayer<MishFunction, MatType>;

/**
 * Standard LiSHT-Layer using the LiSHT activation function.
 */
using LiSHT = BaseLayer<LiSHTFunction, arma::mat>;

template<typename MatType = arma::mat>
using LiSHTType = BaseLayer<LiSHTFunction, MatType>;

/**
 * Standard GELU-Layer using the GELU activation function.
 */
using GELU = BaseLayer<GELUFunction, arma::mat>;

template<typename MatType = arma::mat>
using GELUType = BaseLayer<GELUFunction, MatType>;

/**
 * Standard Elliot-Layer using the Elliot activation function.
 */
using Elliot = BaseLayer<ElliotFunction, arma::mat>;

template<typename MatType = arma::mat>
using ElliotType = BaseLayer<ElliotFunction, MatType>;

/**
 * Standard ELiSH-Layer using the ELiSH activation function.
 */
using Elish = BaseLayer<ElishFunction, arma::mat>;

template<typename MatType = arma::mat>
using ElishType = BaseLayer<ElishFunction, MatType>;

/**
 * Standard Gaussian-Layer using the Gaussian activation function.
 */
using Gaussian = BaseLayer<GaussianFunction, arma::mat>;

template<typename MatType = arma::mat>
using GaussianType = BaseLayer<GaussianFunction, MatType>;

/**
 * Standard HardSwish-Layer using the HardSwish activation function.
 */
using HardSwish = BaseLayer<HardSwishFunction, arma::mat>;

template <typename MatType = arma::mat>
using HardSwishType = BaseLayer<HardSwishFunction, MatType>;

/**
 * Standard TanhExp-Layer using the TanhExp activation function.
 */
using TanhExp = BaseLayer<TanhExpFunction, arma::mat>;

template<typename MatType = arma::mat>
using TanhExpType = BaseLayer<TanhExpFunction, MatType>;

/**
 * Standard SILU-Layer using the SILU activation function.
 */
using SILU = BaseLayer<SILUFunction, arma::mat>;

template<typename MatType = arma::mat>
using SILUType = BaseLayer<SILUFunction, MatType>;

/**
 * Standard Hyper Sinh layer.
 */
using HyperSinh = BaseLayer<HyperSinhFunction, arma::mat>;

template<typename MatType = arma::mat>
using HyperSinhType = BaseLayer<HyperSinhFunction, MatType>;

/**
 * Standard Bipolar Sigmoid layer.
 */
using BipolarSigmoid = BaseLayer<BipolarSigmoidFunction, arma::mat>;

template<typename MatType = arma::mat>
using BipolarSigmoidType = BaseLayer<BipolarSigmoidFunction, MatType>;

} // namespace mlpack

#endif
