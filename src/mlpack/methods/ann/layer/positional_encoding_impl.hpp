/**
 * @file methods/ann/layer/positional_encoding_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Positional Encoding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_POSITIONAL_ENCODING_IMPL_HPP

// In case it hasn't yet been included.
#include "positional_encoding.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
PositionalEncoding<InputDataType, OutputDataType>::PositionalEncoding() :
    embedDim(0),
    maxSequenceLength(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
PositionalEncoding<InputDataType, OutputDataType>::PositionalEncoding(
    const size_t embedDim,
    const size_t maxSequenceLength) :
    embedDim(embedDim),
    maxSequenceLength(maxSequenceLength)
{
  InitPositionalEncoding();
}

template<typename InputDataType, typename OutputDataType>
void PositionalEncoding<InputDataType, OutputDataType>::InitPositionalEncoding()
{
  positionalEncoding.set_size(maxSequenceLength, embedDim);
  const InputDataType position = arma::regspace(0, 1, maxSequenceLength - 1);
  const InputDataType divTerm = arma::exp(arma::regspace(0, 2, embedDim - 1)
      * (- std::log(10000.0) / embedDim));
  const InputDataType theta = position * divTerm.t();
  for (size_t i = 0; i < theta.n_cols; ++i)
  {
    positionalEncoding.col(2 * i) = arma::sin(theta.col(i));
    positionalEncoding.col(2 * i + 1) = arma::cos(theta.col(i));
  }
  positionalEncoding = arma::vectorise(positionalEncoding.t());
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void PositionalEncoding<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  if (input.n_rows != embedDim * maxSequenceLength)
    Log::Fatal << "Incorrect input dimensions!" << std::endl;

  output = input.each_col() + positionalEncoding;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void PositionalEncoding<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  g = gy;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PositionalEncoding<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(embedDim);
  ar & BOOST_SERIALIZATION_NVP(maxSequenceLength);

  if (Archive::is_loading::value)
    InitPositionalEncoding();
}

} // namespace ann
} // namespace mlpack

#endif
