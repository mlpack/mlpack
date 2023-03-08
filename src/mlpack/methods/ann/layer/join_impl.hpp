/**
 * @file methods/ann/layer/join_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Join module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP

// In case it hasn't yet been included.
#include "join.hpp"

namespace mlpack {

template<typename MatType>
JoinType<MatType>::JoinType() :
    Layer<MatType>(),
    inSizeRows(0),
    inSizeCols(0)
{
  // Nothing to do here.
}

template<typename MatType>
JoinType<MatType>::JoinType(const JoinType& other) :
    Layer<MatType>(other),
    inSizeRows(other.inSizeRows),
    inSizeCols(other.inSizeCols)
{
  // Nothing to do here.
}

template<typename MatType>
JoinType<MatType>::JoinType(JoinType&& other) :
    Layer<MatType>(std::move(other)),
    inSizeRows(std::move(other.inSizeRows)),
    inSizeCols(std::move(other.inSizeCols))
{
  // Nothing to do here.
}

template<typename MatType>
JoinType<MatType>&
JoinType<MatType>::operator=(const JoinType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    inSizeRows = other.inSizeRows;
    inSizeCols = other.inSizeCols;
  }

  return *this;
}

template<typename MatType>
JoinType<MatType>&
JoinType<MatType>::operator=(JoinType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    inSizeRows = std::move(other.inSizeRows);
    inSizeCols = std::move(other.inSizeCols);
  }

  return *this;
}

template<typename MatType>
void JoinType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  inSizeRows = input.n_rows;
  inSizeCols = input.n_cols;
  output = arma::vectorise(input);
}

template<typename MatType>
void JoinType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& gy,
    MatType& g)
{
  g = MatType(((MatType&) gy).memptr(), inSizeRows, inSizeCols, false,
      false);
}

template<typename MatType>
template<typename Archive>
void JoinType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(inSizeRows));
  ar(CEREAL_NVP(inSizeCols));
}

} // namespace mlpack

#endif
