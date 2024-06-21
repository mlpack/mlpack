/**
 * @file methods/ann/layer/nearest_interpolation_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the NearestInterpolation layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEAREST_INTERPOLATION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_NEAREST_INTERPOLATION_IMPL_HPP

// In case it hasn't yet been included.
#include "nearest_interpolation.hpp"

namespace mlpack {

template<typename MatType>
NearestInterpolationType<MatType>::NearestInterpolationType():
  Layer<MatType>(),
  inRowSize(0),
  inColSize(0),
  outRowSize(0),
  outColSize(0),
  depth(0),
  batchSize(0)
{
  // Nothing to do here.
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(const size_t inRowSize,
                         const size_t inColSize,
                         const size_t outRowSize,
                         const size_t outColSize,
                         const size_t depth) :
  Layer<MatType>(),
  inRowSize(inRowSize),
  inColSize(inColSize),
  outRowSize(outRowSize),
  outColSize(outColSize),
  depth(depth),
  batchSize(0)
{
  // Nothing to do here.
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(const NearestInterpolationType& other) :
  Layer<MatType>(),
  inRowSize(other.inRowSize),
  inColSize(other.inColSize),
  outRowSize(other.outRowSize),
  outColSize(other.outColSize),
  depth(other.depth),
  batchSize(other.batchSize)
{
  // Nothing to do here. 
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(NearestInterpolationType&& other) :
  Layer<MatType>(std::move(other)),
  inRowSize(std::move(other.inRowSize)),
  inColSize(std::move(other.inColSize)),
  outRowSize(std::move(other.outRowSize)),
  outColSize(std::move(other.outColSize)),
  depth(std::move(other.depth)),
  batchSize(std::move(other.batchSize))
{
  // Nothing to do here.
}

template<typename MatType>
NearestInterpolationType<MatType>&
NearestInterpolationType<MatType>::
operator=(const NearestInterpolationType& other) 
{
  if (&other != this) {
    Layer<MatType>::operator=(other);
    inRowSize = other.inRowSize;
    inColSize = other.inColSize;
    outRowSize = other.outRowSize;
    outColSize = other.outColSize;
    depth = other.depth;
    batchSize = other.batchSize;
  }
  return *this;
}

template<typename MatType>
NearestInterpolationType<MatType>&
NearestInterpolationType<MatType>::
operator=(NearestInterpolationType&& other) 
{
  if (&other != this) {
    Layer<MatType>::operator=(std::move(other));
    inRowSize = std::move(other.inRowSize);
    inColSize = std::move(other.inColSize);
    outRowSize = std::move(other.outRowSize);
    outColSize = std::move(other.outColSize);
    depth = std::move(other.depth);
    batchSize = std::move(other.batchSize);
  }
  return *this;
}

template<typename MatType>
void NearestInterpolationType<MatType>::Forward(
  const MatType& input, MatType& output)
{
  batchSize = input.n_cols;
  if (output.is_empty())
    output.set_size(outRowSize * outColSize * depth, batchSize);
  else
  {
    assert(output.n_rows == outRowSize * outColSize * depth);
    assert(output.n_cols == batchSize);
  }

  assert(inRowSize >= 2);
  assert(inColSize >= 2);

  arma::cube inputAsCube(const_cast<MatType&>(input).memptr(),
      inRowSize, inColSize, depth * batchSize, false, false);
  arma::cube outputAsCube(output.memptr(), outRowSize, outColSize,
      depth * batchSize, false, true);

  double scaleRow = (double) inRowSize / (double) outRowSize;
  double scaleCol = (double) inColSize / (double) outColSize;

  for (size_t i = 0; i < outRowSize; ++i)
  {
    const size_t rOrigin = std::floor(i * scaleRow);

    for (size_t j = 0; j < outColSize; ++j)
    {
      const size_t cOrigin = std::floor(j * scaleCol);

      for (size_t k = 0; k < depth * batchSize; ++k)
      {
        outputAsCube(i, j, k) = inputAsCube.slice(k)(
            rOrigin, cOrigin);
      }
    }
  }
}

template<typename MatType>
void NearestInterpolationType<MatType>::Backward(
  const MatType& /*input*/,
  const MatType& gradient,
  MatType& output)
{
  if (output.is_empty())
  {
    output.zeros(inRowSize * inColSize * depth, batchSize);
  }
  else
  {
    assert(output.n_rows == inRowSize * inColSize * depth);
    assert(output.n_cols == batchSize);
  }

  assert(outRowSize >= 2);
  assert(outColSize >= 2);

  arma::cube outputAsCube(output.memptr(), inRowSize, inColSize,
      depth * batchSize, false, true);
  arma::cube gradientAsCube(((MatType&) gradient).memptr(), outRowSize,
      outColSize, depth * batchSize, false, false);

  double scaleRow = (double)(inRowSize) / outRowSize;
  double scaleCol = (double)(inColSize) / outColSize;

  if (gradient.n_elem == output.n_elem)
  {
    outputAsCube = gradientAsCube;
  }
  else
  {
    for (size_t i = 0; i < outRowSize; ++i)
    {
      const size_t rOrigin = std::floor(i * scaleRow);

      for (size_t j = 0; j < outColSize; ++j)
      {
        const size_t cOrigin = std::floor(j * scaleCol);

        for (size_t k = 0; k < depth * batchSize; ++k)
        {
          outputAsCube(rOrigin, cOrigin, k) +=
              gradientAsCube(i, j, k);
        }
      }
    }
  }
}

template<typename MatType>
void NearestInterpolationType<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions[0] = outRowSize * outColSize * depth;
  this->outputDimensions[1] = batchSize;
}

template<typename MatType>
template<typename Archive>
void NearestInterpolationType<MatType>::serialize(
  Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(inRowSize));
  ar(CEREAL_NVP(inColSize));
  ar(CEREAL_NVP(outRowSize));
  ar(CEREAL_NVP(outColSize));
  ar(CEREAL_NVP(depth));
}

} // namespace mlpack

#endif
