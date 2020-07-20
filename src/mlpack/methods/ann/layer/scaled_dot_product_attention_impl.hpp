/**
 * @file methods/ann/layer/scaled_dot_product_attention_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the ScaledDotProductAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_SCALED_DOT_PRODUCT_ATTENTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SCALED_DOT_PRODUCT_ATTENTION_IMPL_HPP

// In case it hasn't yet been included.
#include "scaled_dot_product_attention.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
ScaledDotProductAttention<InputDataType, OutputDataType>::
ScaledDotProductAttention() :
    embedDim(0),
    dropoutRate(0.0),
    deterministic(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
ScaledDotProductAttention<InputDataType, OutputDataType>::
ScaledDotProductAttention(const size_t embedDim,
    const ElemType dropoutRate,
    const bool deterministic) :
    embedDim(embedDim),
    dropoutRate(dropoutRate),
    deterministic(deterministic)
{
  dropout.Ratio(dropoutRate);
  dropout.Deterministic() = deterministic;
}

template <typename InputDataType, typename OutputDataType>
template <typename eT>
void ScaledDotProductAttention<InputDataType, OutputDataType>::
Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  /*
  embedding dimension: embedDim
  target sequence length: (input.n_rows / embedDim)
  source sequence length: (key.n_rows / embedDim)
  batch size: input.n_cols
  */
  typedef typename arma::Cube<eT> CubeType;

  Log::Assert(input.n_rows % embedDim == 0,
      "Number of features in input must be divisible by embedding dimension.");

  CubeType q, k, v;
  Expand(input, key, value, q, k, v);

  q /= std::sqrt(embedDim);
  CubeType scores = CubeMultiply(k, q, true, false);

  if (!attnMask.is_empty())
  {
    if (attnMask.n_rows != k.n_cols || attnMask.n_cols != q.n_cols)
    {
      Log::Fatal << "The size of the 2D `attn_mask` is not correct."
                 << std::endl;
    }
    scores.each_slice() += attnMask;
  }
  if (!keyPaddingMask.is_empty())
  {
    if (keyPaddingMask.n_rows != k.n_cols || keyPaddingMask.n_cols != 1)
    {
      Log::Fatal << "The size of the `keyPaddingMask` is not valid."
                 << std::endl;
    }
    scores.each_slice() += arma::repmat(keyPaddingMask, 1, q.n_cols);
  }

  attnOut.set_size(k.n_cols, q.n_cols, input.n_cols);
  softmaxOutput.set_size(k.n_cols, q.n_cols, input.n_cols);

  for (size_t i = 0; i < input.n_cols; ++i)
  {
    softmax.Forward(scores.slice(i), softmaxOutput.slice(i));
    dropout.Forward(softmaxOutput.slice(i), attnOut.slice(i));
  }

  scores = CubeMultiply(v, attnOut, false, false);

  output.set_size(embedDim * q.n_cols, input.n_cols);
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    output.col(i) = arma::vectorise(scores.slice(i));
  }
}

template <typename InputDataType, typename OutputDataType>
template <typename eT>
void ScaledDotProductAttention<InputDataType, OutputDataType>::
Backward(const arma::Mat<eT>& input,
         const arma::Mat<eT>& gy,
         arma::Mat<eT>& g)
{
  typedef typename arma::Mat<eT> MatType;
  typedef typename arma::Cube<eT> CubeType;

  CubeType q, k, v;
  Expand(input, key, value, q, k, v);

  CubeType gyTemp(embedDim, gy.n_rows / embedDim, gy.n_cols);
  for (size_t i = 0; i < gy.n_elem; ++i)
  {
    const size_t row = i % embedDim;
    const size_t col = (i / embedDim) % (gy.n_rows / embedDim);
    const size_t slice = i / gy.n_rows;
    gyTemp(row, col, slice) = gy(i);
  }

  gyTemp = CubeMultiply(v, gyTemp, true, false);

  for (size_t i = 0; i < gy.n_cols; ++i)
  {
    dropout.Backward(MatType(), gyTemp.slice(i), dropout.Delta());
    softmax.Backward(softmaxOutput.slice(i), dropout.Delta(), gyTemp.slice(i));
  }

  gyTemp = CubeMultiply(k, gyTemp) / std::sqrt(embedDim);

  g.set_size(arma::size(gy));
  for (size_t i = 0; i < gy.n_cols; ++i)
  {
    g.submat(0, i, g.n_rows - 1, i) = arma::vectorise(gyTemp.slice(i));
  }
}

template <typename InputDataType, typename OutputDataType>
template <typename Archive>
void ScaledDotProductAttention<InputDataType, OutputDataType>::
serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(embedDim);
  ar & BOOST_SERIALIZATION_NVP(dropout);
  ar & BOOST_SERIALIZATION_NVP(deterministic);
}

} // namespace ann
} // namespace mlpack

#endif
