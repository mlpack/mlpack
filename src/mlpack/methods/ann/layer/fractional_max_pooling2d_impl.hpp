/**
 * @file methods/ann/layer/fractional_max_pooling2d_impl.hpp
 * @author Mayank Raj
 *
 * Implementation of the FractionalMaxPooling2D class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_FRACTIONAL_MAX_POOLING2D_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_FRACTIONAL_MAX_POOLING2D_IMPL_HPP

// In case it hasn't yet been included.
#include "fractional_max_pooling2d.hpp"

namespace mlpack {

template<typename MatType>
FractionalMaxPooling2DType<MatType>::FractionalMaxPooling2DType():
    Layer<MatType>()
{
}

template<typename MatType>
FractionalMaxPooling2DType<MatType>::FractionalMaxPooling2DType(const double poolingRatio)
  : Layer<MatType>(),
  poolingRatio(poolingRatio)
{
  if (poolingRatio <= 0)
    throw std::invalid_argument("poolingRatio must be greater than 0.0");
}

template<typename MatType>
FractionalMaxPooling2DType<MatType>::FractionalMaxPooling2DType(const FractionalMaxPooling2DType& other)
  : Layer<MatType>(other),
  poolingRatio(other.poolingRatio)
{ }


template<typename MatType>
FractionalMaxPooling2DType<MatType>& FractionalMaxPooling2DType<MatType>::operator=(const FractionalMaxPooling2DType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    poolingRatio = other.poolingRatio;
  }
  return *this;
}

template<typename MatType>
FractionalMaxPooling2DType<MatType>& FractionalMaxPooling2DType<MatType>::operator=(FractionalMaxPooling2DType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    poolingRatio = std::move(other.poolingRatio);
  }
  return *this;
}

template<typename MatType>
void FractionalMaxPooling2DType<MatType>::Forward(const MatType& input, MatType& output)
{
  // Calculate new dimensions based on the pooling ratio.
  size_t outputRows = static_cast<size_t>(std::ceil(input.n_rows / poolingRatio));
  size_t outputCols = static_cast<size_t>(std::ceil(input.n_cols / poolingRatio));

  output.set_size(outputRows, outputCols);

  for (size_t i = 0; i < output.n_rows; ++i)
  {
    for (size_t j = 0; j < output.n_cols; ++j)
    {
      size_t startRow = i * poolingRatio;
      size_t endRow = std::min(static_cast<size_t>(input.n_rows), static_cast<size_t>(startRow + poolingRatio));
      size_t startCol = j * poolingRatio;
      size_t endCol = std::min(static_cast<size_t>(input.n_cols), static_cast<size_t>(startCol + poolingRatio));

      output(i, j) = input.submat(startRow, startCol, endRow - 1, endCol - 1).max();
    }
  }
}

template<typename MatType>
void FractionalMaxPooling2DType<MatType>::Backward(const MatType& input,
                const MatType& gy,
                MatType& g)
{
  g.set_size(input.n_rows, input.n_cols);
  g.zeros();
  for (size_t i = 0; i < gy.n_rows; ++i)
  {
    for (size_t j = 0; j < gy.n_cols; ++j)
    {
      size_t startRow = i * poolingRatio;
      size_t endRow = std::min(static_cast<size_t>(input.n_rows), static_cast<size_t>(startRow + poolingRatio));
      size_t startCol = j * poolingRatio;
      size_t endCol = std::min(static_cast<size_t>(input.n_cols), static_cast<size_t>(startCol + poolingRatio));

      g.submat(startRow, startCol, endRow - 1, endCol - 1).fill(gy(i, j) / (poolingRatio * poolingRatio));
    }
  }
}

template<typename MatType>
template<typename Archive>
void FractionalMaxPooling2DType<MatType>::serialize(Archive& ar, const uint32_t version)
{
  ar(CEREAL_NVP(poolingRatio));
  // Serialize other member variables if necessary.
}

} // namespace mlpack

#endif
