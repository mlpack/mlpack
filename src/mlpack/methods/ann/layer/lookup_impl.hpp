/**
 * @file methods/ann/layer/lookup_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Lookup class a particular convolution, where the width
 * of the convolution is 1.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP

// In case it hasn't yet been included.
#include "lookup.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
Lookup<InputDataType, OutputDataType>::Lookup(
    const size_t inSize,
    const size_t outSize) :
    inSize(inSize),
    outSize(outSize)
{
  weights.set_size(outSize, inSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  Log::Assert((size_t) input.n_rows % inSize == 0);

  const size_t seqLength = input.n_rows / inSize;
  const size_t batchSize = input.n_cols;

  arma::Cube<eT> inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(), inSize,
      inSize, seqLength, batchSize, true, false);

  output.set_size(outSize * seqLength, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
    output.col(i) = arma::vectorise(weights.cols(
        arma::conv_to<arma::uvec>::from(inputTemp.slice(i)) - 1));
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = gy;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  Log::Assert((size_t) input.n_rows % inSize == 0);

  const size_t seqLength = input.n_rows / inSize;
  const size_t batchSize = input.n_cols;

  arma::Cube<eT> inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(), inSize,
      seqLength, batchSize, true, false);
  arma::Cube<eT> errorTemp(const_cast<arma::Mat<eT>&>(error).memptr(), outSize,
      seqLength, batchSize, true, false);

  arma::Cube<eT> dW(weights.n_rows, weights.n_cols, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
  dW.slice(i).cols(arma::conv_to<arma::uvec>::from(inputTemp.slice(i)) - 1)
      = errorTemp.slice(i);
  }

  gradient = arma::mean(dW, 2);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Lookup<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
    weights.set_size(outSize, inSize);
}

} // namespace ann
} // namespace mlpack

#endif
