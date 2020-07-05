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
ScaledDotProductAttention(const size_t targetLength,
    const size_t sourceLength,
    const size_t embedDim,
    const ElemType dropoutRate,
    const bool deterministic) :
    targetLength(targetLength),
    sourceLength(sourceLength),
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
  typedef typename arma::Cube<eT> CubeType;
  typedef typename arma::Mat<eT> MatType;

  Log::Assert(input.n_rows == embedDim * (targetLength + 2 * sourceLength));

  const size_t batchSize = input.n_cols;
  const size_t qStart = 0, qEnd = embedDim * targetLength - 1;
  const size_t kStart = qEnd + 1, kEnd = kStart + embedDim * sourceLength - 1;
  const size_t vStart = kEnd + 1, vEnd = vStart + embedDim * sourceLength - 1;
  output.set_size(embedDim * targetLength, batchSize);

  MatType q = input.submat(qStart, 0, qEnd, batchSize - 1);
  MatType k = input.submat(kStart, 0, kEnd, batchSize - 1);
  MatType v = input.submat(vStart, 0, vEnd, batchSize - 1);

  CubeType query(q.memptr(), embedDim, targetLength, batchSize, false, false);
  CubeType key(k.memptr(), embedDim, sourceLength, batchSize, false, false);
  CubeType value(v.memptr(), embedDim, sourceLength, batchSize, false, false);

  query /= std::sqrt(embedDim);
  CubeType scores = CubeMultiply(key, query, true, false);

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

  attnOut.set_size(sourceLength, targetLength, batchSize);
  softmaxOutput.set_size(sourceLength, targetLength, batchSize);
  for (size_t i = 0; i < batchSize; ++i)
  {
    softmax.Forward(scores.slice(i), softmax.OutputParameter());
    softmaxOutput.slice(i) = softmax.OutputParameter();
    dropout.Forward(softmax.OutputParameter(), attnOut.slice(i));
  }
  scores = CubeMultiply(value, attnOut, false, false);
  for (size_t i = 0; i < batchSize; ++i)
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
  typedef typename arma::Cube<eT> CubeType;
  typedef typename arma::Mat<eT> MatType;

  const size_t batchSize = gy.n_cols;
  const size_t qStart = 0, qEnd = embedDim * targetLength - 1;
  const size_t kStart = qEnd + 1, kEnd = kStart + embedDim * sourceLength - 1;
  const size_t vStart = kEnd + 1, vEnd = vStart + embedDim * sourceLength - 1;
  g.set_size(embedDim * (targetLength + 2 * sourceLength), batchSize);

  MatType q = input.submat(qStart, 0, qEnd, batchSize - 1);
  MatType k = input.submat(kStart, 0, kEnd, batchSize - 1);
  MatType v = input.submat(vStart, 0, vEnd, batchSize - 1);

  CubeType query(q.memptr(), embedDim, targetLength, batchSize, false, false);
  CubeType key(k.memptr(), embedDim, sourceLength, batchSize, false, false);
  CubeType value(v.memptr(), embedDim, sourceLength, batchSize, false, false);

  CubeType gy3d(const_cast<MatType&>(gy).memptr(),
      embedDim, targetLength, batchSize, false, false);
  CubeType gyTemp = CubeMultiply(value, gy3d, true, false);

  for (size_t i = 0; i < batchSize; ++i)
  {
    dropout.Backward(MatType(), gyTemp.slice(i), dropout.Delta());
    softmax.Backward(softmaxOutput.slice(i), dropout.Delta(), gyTemp.slice(i));
  }

  gyTemp /= std::sqrt(embedDim);
  CubeType gQuery = CubeMultiply(key, gyTemp);
  CubeType gKey = CubeMultiply(gyTemp, query, false, true);
  CubeType gValue = CubeMultiply(gy3d, attnOut, false, true);
  for (size_t i = 0; i < batchSize; ++i)
  {
    g.submat(qStart, i, qEnd, i) = arma::vectorise(gQuery.slice(i));
    g.submat(kStart, i, kEnd, i) = arma::vectorise(gKey.slice(i));
    g.submat(vStart, i, vEnd, i) = arma::vectorise(gValue.slice(i));
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
