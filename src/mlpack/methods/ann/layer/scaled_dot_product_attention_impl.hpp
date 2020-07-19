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
    targetLength(0),
    sourceLength(0),
    embedDim(0),
    dropoutRate(0.0),
    deterministic(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
ScaledDotProductAttention<InputDataType, OutputDataType>::
ScaledDotProductAttention(const size_t embedDim,
    const InputDataType& key,
    const InputDataType& value,
    const ElemType dropoutRate,
    const bool deterministic) :
    embedDim(embedDim),
    key(key),
    value(value),
    dropoutRate(dropoutRate),
    deterministic(deterministic)
{
  if (!this->key.is_empty() && this->value.is_empty())
    this->value = this->key;

  if (!key.is_empty() && !value.is_empty())
  {
    if (key.n_rows != value.n_rows || this->key.n_cols != value.n_cols)
    {
      Log::Fatal << "The 'key' and 'value' matrices must have the same "
                 << "dimensions.";
    }
  }
  Log::Assert(key.n_rows % embedDim == 0);
  sourceLength = key.n_rows / embedDim;
  dropout.Ratio(dropoutRate);
  dropout.Deterministic() = deterministic;
}

template <typename InputDataType, typename OutputDataType>
template <typename eT>
void ScaledDotProductAttention<InputDataType, OutputDataType>::
Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  Log::Assert(input.n_rows % embedDim == 0);
  targetLength = input.n_rows / embedDim;

  if (key.is_empty())
  {
    key = const_cast<arma::Mat<eT>&>(input);
    value = const_cast<arma::Mat<eT>&>(input);
  }
  arma::Cube<eT> q(const_cast<arma::Mat<eT>&>(input).memptr(),
      embedDim, targetLength, input.n_cols, true, false);
  arma::Cube<eT> k(key.memptr(), embedDim, sourceLength, input.n_cols, false, false);
  arma::Cube<eT> v(value.memptr(), embedDim, sourceLength, input.n_cols, false, false);

  q /= std::sqrt(embedDim);
  arma::Cube<eT> scores = CubeMultiply(k, q, true, false);

  if (!attnMask.is_empty())
  {
    if (attnMask.n_rows != sourceLength || attnMask.n_cols != targetLength)
    {
      Log::Fatal << "The size of the 2D `attn_mask` is not correct."
                 << std::endl;
    }
    scores.each_slice() += attnMask;
  }
  if (!keyPaddingMask.is_empty())
  {
    if (keyPaddingMask.n_rows != sourceLength || keyPaddingMask.n_cols != 1)
    {
      Log::Fatal << "The size of the `keyPaddingMask` is not correct."
                 << std::endl;
    }
    scores.each_slice() += arma::repmat(keyPaddingMask, 1, targetLength);
  }

  attnOut.set_size(sourceLength, targetLength, input.n_cols);
  softmaxOutput.set_size(sourceLength, targetLength, input.n_cols);
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    softmax.Forward(scores.slice(i), softmax.OutputParameter());
    softmaxOutput.slice(i) = softmax.OutputParameter();
    dropout.Forward(softmax.OutputParameter(), attnOut.slice(i));
  }
  scores = CubeMultiply(v, attnOut, false, false);
  output.set_size(embedDim * targetLength, input.n_cols);
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    output.col(i) = arma::vectorise(scores.slice(i));
  }
}

template <typename InputDataType, typename OutputDataType>
template <typename eT>
void ScaledDotProductAttention<InputDataType, OutputDataType>::
Backward(const arma::Mat<eT>& /* input */,
         const arma::Mat<eT>& gy,
         arma::Mat<eT>& g)
{
  g.set_size(arma::size(gy));
  arma::Cube<eT> k(key.memptr(), embedDim, sourceLength, gy.n_cols, false, false);
  arma::Cube<eT> v(value.memptr(), embedDim, sourceLength, gy.n_cols, false, false);

  arma::Cube<eT> gyTemp(const_cast<arma::Mat<eT>&>(gy).memptr(), embedDim,
      targetLength, gy.n_cols, true, false);

  gyTemp = CubeMultiply(v, gyTemp, true, false);

  for (size_t i = 0; i < gy.n_cols; ++i)
  {
    dropout.Backward(arma::Mat<eT>(), gyTemp.slice(i), dropout.Delta());
    softmax.Backward(softmaxOutput.slice(i), dropout.Delta(), gyTemp.slice(i));
  }

  gyTemp = CubeMultiply(k, gyTemp) / std::sqrt(embedDim);

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
  ar & BOOST_SERIALIZATION_NVP(targetLength);
  ar & BOOST_SERIALIZATION_NVP(sourceLength);
  ar & BOOST_SERIALIZATION_NVP(embedDim);
  ar & BOOST_SERIALIZATION_NVP(dropout);
  ar & BOOST_SERIALIZATION_NVP(deterministic);
}

} // namespace ann
} // namespace mlpack

#endif
