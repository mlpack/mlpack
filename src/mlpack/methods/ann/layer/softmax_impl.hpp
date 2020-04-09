/**
 * @file softmax_impl.hpp
 * @author Mrityunjay Tripathi
 * @author Sreenik Seal
 *
 * Implementation of the Softmax class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP

// In case it hasn't yet been included.
#include "softmax.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SoftMax<InputDataType, OutputDataType>::SoftMax() : deterministic(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void SoftMax<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output.set_size(arma::size(input));

  InputType inputMax = arma::repmat(arma::max(input, 0), input.n_rows, 1);

  output = inputMax + arma::log(arma::repmat(
      arma::sum(arma::exp(input - inputMax), 0), input.n_rows, 1));
  output = arma::exp(input - output);

  if (!deterministic)
  {
    y = output;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void SoftMax<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = y % (gy - arma::repmat(gy.t() * y, gy.n_rows, gy.n_cols));
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SoftMax<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
