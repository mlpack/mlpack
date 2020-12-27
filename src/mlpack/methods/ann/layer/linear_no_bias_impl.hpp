/**
 * @file methods/ann/layer/linear_no_bias_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LinearNoBias class also known as fully-connected layer
 * or affine transformation without the bias term.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP

// In case it hasn't yet been included.
#include "linear_no_bias.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType, typename RegularizerType>
LinearNoBiasType<InputType, OutputType, RegularizerType>::LinearNoBiasType() :
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType, typename RegularizerType>
LinearNoBiasType<InputType, OutputType, RegularizerType>::LinearNoBiasType(
    const size_t inSize,
    const size_t outSize,
    RegularizerType regularizer) :
    inSize(inSize),
    outSize(outSize),
    regularizer(regularizer)
{
  weights.set_size(outSize * inSize, 1);
}

template<typename InputType, typename OutputType, typename RegularizerType>
void LinearNoBiasType<InputType, OutputType, RegularizerType>::Reset()
{
  weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
}

template<typename InputType, typename OutputType, typename RegularizerType>
void LinearNoBiasType<InputType, OutputType, RegularizerType>::Forward(
    const InputType& input, OutputType& output)
{
  output = weight * input;
}

template<typename InputType, typename OutputType, typename RegularizerType>
void LinearNoBiasType<InputType, OutputType, RegularizerType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  g = weight.t() * gy;
}

template<typename InputType, typename OutputType, typename RegularizerType>
void LinearNoBiasType<InputType, OutputType, RegularizerType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
      error * input.t());
  regularizer.Evaluate(weights, gradient);
}

template<typename InputType, typename OutputType, typename RegularizerType>
template<typename Archive>
void LinearNoBiasType<InputType, OutputType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));

  // This is inefficient, but necessary so that WeightSetVisitor sets the right
  // size.
  if (cereal::is_loading<Archive>())
    weights.set_size(outSize * inSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif
