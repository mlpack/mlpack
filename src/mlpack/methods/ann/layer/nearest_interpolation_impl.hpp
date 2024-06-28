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
  Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(const double scaleFactor) :
  Layer<MatType>()
{
  scaleFactors = std::vector<double>(2);
  scaleFactors[0] = scaleFactor;
  scaleFactors[1] = scaleFactor;
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(const std::vector<double> scaleFactors) :
  Layer<MatType>(),
  scaleFactors(scaleFactors)
{
  // Nothing to do here.
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(const NearestInterpolationType& other) :
  Layer<MatType>(),
  scaleFactors(other.scaleFactors)
{
  // Nothing to do here. 
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(NearestInterpolationType&& other) :
  Layer<MatType>(std::move(other)),
  scaleFactors(std::move(other.scaleFactors))
{
  // Nothing to do here.
}

template<typename MatType>
NearestInterpolationType<MatType>&
NearestInterpolationType<MatType>::
operator=(const NearestInterpolationType& other) 
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    scaleFactors = other.scaleFactors;
  }
  return *this;
}

template<typename MatType>
NearestInterpolationType<MatType>&
NearestInterpolationType<MatType>::
operator=(NearestInterpolationType&& other) 
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    scaleFactors = std::move(other.scaleFactors);
  }
  return *this;
}

template<typename MatType>
void NearestInterpolationType<MatType>::Forward(
  const MatType& input, MatType& output)
{
  size_t channels = this->inputDimensions[0];

  size_t outRowSize = this->outputDimensions[1];
  size_t outColSize = this->outputDimensions[2];

  size_t inRowSize = this->inputDimensions[1];
  size_t inColSize = this->inputDimensions[2];

  assert(output.n_rows == channels);
  assert(output.n_cols == outRowSize * outColSize);

  arma::cube inputAsCube;
  arma::cube outputAsCube;

  MakeAlias(inputAsCube, input, channels, inRowSize, inColSize, 0, false);
  MakeAlias(outputAsCube, output, channels, outRowSize, outColSize, 0, true);

  for (size_t i = 0; i < channels; ++i)
  {
    for (size_t j = 0; j < outRowSize; ++j)
    {
      size_t rOrigin = std::floor(j * 1.0f / scaleFactors[0]);
      for (size_t k = 0; k < outColSize; ++k)
      {
        size_t cOrigin = std::floor(k * 1.0f / scaleFactors[1]);

        outputAsCube(i, j, k) = inputAsCube(i, rOrigin, cOrigin);
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
  size_t channels = this->inputDimensions[0];

  size_t outRowSize = this->outputDimensions[1];
  size_t outColSize = this->outputDimensions[2];

  size_t inRowSize = this->inputDimensions[1];
  size_t inColSize = this->inputDimensions[2];

  assert(output.n_rows == channels);
  assert(output.n_cols == inRowSize * inColSize);

  arma::cube outputAsCube;
  arma::cube gradientAsCube;

  MakeAlias(outputAsCube, output, channels, inRowSize, inColSize, 0, true);
  MakeAlias(gradientAsCube, gradient, channels, outRowSize, outColSize, 0, false);

  for (size_t i = 0; i < channels; ++i)
  {
    for (size_t j = 0; j < outRowSize; ++j)
    {
      size_t rOrigin = std::floor(j * 1.0f / scaleFactors[0]);
      for (size_t k = 0; k < outColSize; ++k)
      {
        size_t cOrigin = std::floor(k * 1.0f / scaleFactors[1]);
        outputAsCube(i, rOrigin, cOrigin) += gradientAsCube(i, j, k);
      }
    }
  }
}

template<typename MatType>
void NearestInterpolationType<MatType>::ComputeOutputDimensions()
{
  assert(this->inputDimensions.size() - 1 == scaleFactors.size());
  this->outputDimensions = this->inputDimensions;
  for (size_t i = 1; i < this->InputDimensions().size(); i++)
  {
    this->outputDimensions[i] = std::round((double)this->outputDimensions[i] * scaleFactors[i-1]);
  }
}

template<typename MatType>
template<typename Archive>
void NearestInterpolationType<MatType>::serialize(
  Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(scaleFactors));
}

} // namespace mlpack

#endif
