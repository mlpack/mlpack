/**
 * @file methods/ann/layer/scalar_dot_product_attention_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the ScalarDotProductAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_SCALAR_DOT_PRODUCT_ATTENTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SCALAR_DOT_PRODUCT_ATTENTION_IMPL_HPP

// In case it hasn't yet been included.
#include "scalar_dot_product_attention.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
ScalarDotProductAttention<InputDataType, OutputDataType>::
ScalarDotProductAttention() :
    tLen(0),
    sLen(0),
    embedDim(0),
    dropoutRate(0.0),
    deterministic(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
ScalarDotProductAttention<InputDataType, OutputDataType>::
ScalarDotProductAttention(const size_t tLen,
    const size_t sLen,
    const size_t embedDim,
    const ElemType dropoutRate,
    const bool deterministic) :
    tLen(tLen),
    sLen(sLen),
    embedDim(embedDim),
    dropoutRate(dropoutRate),
    deterministic(deterministic)
{
  dropout.Ratio(dropoutRate);
  dropout.Deterministic() = deterministic;
}

template <typename InputDataType, typename OutputDataType>
template <typename eT>
void ScalarDotProductAttention<InputDataType, OutputDataType>::
Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  typedef typename arma::Cube<eT> CubeType;
  typedef typename arma::Mat<eT> MatType;

  Log::Assert(input.n_rows == embedDim * (tLen + 2 * sLen));

  const size_t bsz = input.n_cols;
  output.set_size(embedDim * tLen, bsz);

  CubeType query(const_cast<MatType&>(input).memptr(),
      embedDim, tLen, bsz, true, false);
  CubeType key(const_cast<MatType&>(input).memptr()
      + query.n_elem, embedDim, sLen, bsz, true, false);
  CubeType value(const_cast<MatType&>(input).memptr()
      + query.n_elem + key.n_elem, embedDim, sLen, bsz, true, false);

  query /= std::sqrt(embedDim);
  CubeType scores = CubeMultiply(key, value, 1, 0);

  if (!attnMask.is_empty())
  {
    if (attnMask.n_rows != sLen || attnMask.n_cols != tLen)
    {
      Log::Fatal << "The size of the 2D `attn_mask` is not correct."
                 << std::endl;
    }
    scores.each_slice() += attnMask;
  }

  if (!keyPaddingMask.is_empty())
  {
    if (keyPaddingMask.n_rows != sLen || keyPaddingMask.n_cols != 1)
    {
      Log::Fatal << "The size of the `keyPaddingMask` is not correct."
                 << std::endl;
    }
    scores.each_slice() += arma::repmat(keyPaddingMask, 1, tLen);
  }

  MatType Wt2d(scores.memptr(), sLen * tLen, bsz, 0, 0);

  softmax.Forward(Wt2d, softmax.OutputParameter());
  dropout.Forward(softmax.OutputParameter(), dropout.OutputParameter());
  Wt2d = dropout.OutputParameter();

  attnOut = CubeType(Wt2d.memptr(), sLen, tLen, bsz, 0, 0);
  scores = CubeMultiply(value, attnOut, 0, 0);
  // attnOut = CubeType(attnOut.memptr(), embedDim, tLen, bsz, 0, 0);
  for (size_t i = 0; i < bsz; ++i)
  {
    output.col(i) = arma::vectorise(scores.slice(i));
  }
}

template <typename InputDataType, typename OutputDataType>
template <typename eT>
void ScalarDotProductAttention<InputDataType, OutputDataType>::
Backward(const arma::Mat<eT>& input,
         const arma::Mat<eT>& gy,
         arma::Mat<eT>& g)
{
  typedef typename arma::Cube<eT> CubeType;
  typedef typename arma::Mat<eT> MatType;

  const size_t bsz = gy.n_cols;
  g.set_size(embedDim * (tLen + 2 * sLen), bsz);

  CubeType query(const_cast<MatType&>(input).memptr(),
      embedDim, tLen, bsz, true, false);
  CubeType key(const_cast<MatType&>(input).memptr()
      + query.n_elem, embedDim, sLen, bsz, true, false);
  CubeType value(const_cast<MatType&>(input).memptr()
      + query.n_elem + key.n_elem, embedDim, sLen, bsz, true, false);

  CubeType gy3d(const_cast<MatType&>(gy).memptr(), embedDim, tLen, bsz, 0, 0);
  CubeType gyTemp = CubeMultiply(value, gy3d, true, false);

  dropout.Backward(MatType(), MatType(gyTemp.memptr(),
      sLen * tLen, bsz, 0, 0),  dropout.Delta());
  softmax.Backward(softmax.OutputParameter(), dropout.Delta(), softmax.Delta());

  gyTemp = CubeType(softmax.Delta().memptr(), sLen, tLen, bsz, 0, 0);

  if (!attnMask.is_empty())
    gyTemp.each_slice() += attnMask;
  if (!keyPaddingMask.is_empty())
    gyTemp.each_slice() += arma::repmat(keyPaddingMask, 1, tLen);

  gyTemp /= std::sqrt(embedDim);
  CubeType gQuery = CubeMultiply(key, gyTemp);
  CubeType gKey = CubeMultiply(gyTemp, query, 0, 1);
  CubeType gValue = CubeMultiply(gy3d, attnOut, 0, 1);
  for (size_t i = 0; i < bsz; ++i)
  {
    g.submat(0, i, embedDim * tLen - 1, i) = arma::vectorise(gQuery.slice(i));
    g.submat(embedDim * tLen, i, embedDim * (tLen + sLen) - 1, i)
        = arma::vectorise(gKey.slice(i));
    g.submat(embedDim * (tLen + sLen), i, embedDim * (tLen + 2 * sLen) - 1, i)
        = arma::vectorise(gValue.slice(i));
  }
}

template <typename InputDataType, typename OutputDataType>
template <typename Archive>
void ScalarDotProductAttention<InputDataType, OutputDataType>::
serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(tLen);
  ar & BOOST_SERIALIZATION_NVP(sLen);
  ar & BOOST_SERIALIZATION_NVP(embedDim);
  ar & BOOST_SERIALIZATION_NVP(dropout);
  ar & BOOST_SERIALIZATION_NVP(deterministic);
}

} // namespace ann
} // namespace mlpack

#endif