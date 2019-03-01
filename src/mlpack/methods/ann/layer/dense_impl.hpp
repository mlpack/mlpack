/**
 * @file dense_impl.hpp
 * @author N Rajiv Vaidyanathan
 *
 * Implementation of the Dense block class, which improves gradient and feature
 * propogation. Reduces the number of parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DENSE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_DENSE_IMPL_HPP

#include "batch_norm.hpp"
#include "leaky_relu.hpp"
#include "convolution.hpp"
#include "dropout.hpp"

// In case it hasn't yet been included.
#include "dense.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Dense<InputDataType, OutputDataType>::Dense()
{
  // Nothing to do here.
}
template<typename InputDataType, typename OutputDataType>
Dense<InputDataType, OutputDataType>::Dense(const size_t nb_layers,
    const size_t growth_rate, const bool bottleneck,
    const double dropout_rate, const double weight_decay) :
    nb_layers(nb_layers),
    growth_rate(growth_rate),
    bottleneck(bottleneck),
    dropout_rate(dropout_rate),
    weight_decay(weight_decay)
{
  // Nothing to do here.
}

template<typename eT>
arma::cube conv_block(arma::Cube<eT>&& input, const size_t input_size,
    const size_t growth_rate, const bool bottleneck,
    const double dropout_rate, const double weight_decay)
{
  BatchNorm<> bn(input.n_rows);
  bn.Reset();
  bn.Deterministic() = false;
  bn.Forward(std::move(input), input);

  LeakyReLU<> l(0.0);
  l.Forward(input, input);

  if (bottleneck)
  {
    Convolution<> b(input_size, growth_rate, 1, 1, 1, 1, 0, 0,
    input.n_cols, input.n_rows);
    b.Forward(input, input);

    bn.Forward(std::move(input), input);

    l.Forward(input, input);
  }

  Convolution<> c(input_size, growth_rate, 3, 3, 1, 1, 0, 0,
  input.n_cols, input.n_rows);
  c.Forward(input, input);

  if (dropout_rate)
  {
    Dropout<> d(dropout_rate);
    d.Forward(input, input);
  }

  return input;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Dense<InputDataType, OutputDataType>::Forward(
    const arma::Cube<eT>&& input, arma::Cube<eT>&& output)
{
  output = input;
  for (size_t i = 0; i < nb_layers; i++)
  {
    // Create a conv_block
    arma::cube cb = conv_block(output, input.n_slices, growth_rate,
    bottleneck, dropout_rate, weight_decay);
    // Concatenated with the current input
    output = arma::join_slices(output, cb);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Dense<InputDataType, OutputDataType>::Backward(
    const arma::Cube<eT>&& input, arma::Cube<eT>&& gy, arma::Cube<eT>&& g)
{
  // Yet to be implemented.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Dense<InputDataType, OutputDataType>::Gradient(
    const arma::Cube<eT>&& /* input */,
    arma::Cube<eT>&& error,
    arma::Cube<eT>&& gradient)
{
  // Yet to be implemented.
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Dense<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  // Yet to be implemented.
}

} // namespace ann
} // namespace mlpack

#endif
