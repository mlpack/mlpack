/**
 * @file methods/ann/layer/parametric_relu_impl.hpp
 * @author Prasanna Patil
 *
 * Definition of PReLU layer first introduced in the,
 * Kaiming He, Xiangyu Zhang, Shaoqing, Ren Jian Sun,
 * "Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification", 2014
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PRELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_PRELU_IMPL_HPP

// In case it hasn't yet been included.
#include "parametric_relu.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
PReLUType<InputType, OutputType>::PReLUType(
    const double userAlpha) : userAlpha(userAlpha)
{
  alpha.set_size(WeightSize(), 1);
  alpha(0) = userAlpha;
}

template<typename InputType, typename OutputType>
void PReLUType<InputType, OutputType>::SetWeights(
    typename OutputType::elem_type* weightsPtr)
{
  alpha = arma::mat(weightsPtr, 1, 1, false, false);

  //! Set value of alpha to the one given by user.
  // TODO: this doesn't even make any sense.  is it trainable or not?
  // why is there userAlpha?  is that for initialization only?
  alpha(0) = userAlpha;
}

template<typename InputType, typename OutputType>
void PReLUType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  // TODO: use transform()?
  output = input;
  arma::uvec negative = arma::find(input < 0);
  output(negative) = input(negative) * alpha(0);
}

template<typename InputType, typename OutputType>
void PReLUType<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  OutputType derivative;
  derivative.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
    derivative(i) = (input(i) >= 0) ? 1 : alpha(0);

  g = gy % derivative;
}

template<typename InputType, typename OutputType>
void PReLUType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  OutputType zeros = arma::zeros<OutputType>(input.n_rows, input.n_cols);
  gradient(0) = arma::accu(error % arma::min(zeros, input)) / input.n_cols;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void PReLUType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif
