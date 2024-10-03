/**
 * @file methods/ann/layer/nearest_interpolation_impl.hpp
 * @author Andrew Furey
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
#include <stdexcept>

namespace mlpack {

template<typename MatType>
NearestInterpolationType<MatType>::NearestInterpolationType():
  Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
NearestInterpolationType<MatType>::
NearestInterpolationType(const std::vector<double> scaleFactors) :
  Layer<MatType>()
{
  if (scaleFactors.size() != 2) {
    throw std::runtime_error("Scale factors must have 2 dimensions");
  }
  this->scaleFactors = std::move(scaleFactors);
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
  const size_t channels = this->inputDimensions[2];

  const size_t outRowSize = this->outputDimensions[0];
  const size_t outColSize = this->outputDimensions[1];

  const size_t inRowSize = this->inputDimensions[0];
  const size_t inColSize = this->inputDimensions[1];

  arma::cube inputAsCube;
  arma::cube outputAsCube;

  MakeAlias(inputAsCube, input, inRowSize, inColSize, channels, 0, false);
  MakeAlias(outputAsCube, output, outRowSize, outColSize, channels, 0, true);

  for (size_t i = 0; i < outRowSize; ++i)
  {
    size_t rOrigin = std::floor(i  / scaleFactors[0]);
    for (size_t j = 0; j < outColSize; ++j)
    {
      size_t cOrigin = std::floor(j / scaleFactors[1]);
      for (size_t k = 0; k < channels; ++k)
      {
        outputAsCube(i, j, k) = inputAsCube(rOrigin, cOrigin, k);
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
  const size_t channels = this->inputDimensions[2];

  const size_t outRowSize = this->outputDimensions[0];
  const size_t outColSize = this->outputDimensions[1];

  const size_t inRowSize = this->inputDimensions[0];
  const size_t inColSize = this->inputDimensions[1];

  arma::cube outputAsCube;
  arma::cube gradientAsCube;

  MakeAlias(outputAsCube, output, inRowSize, inColSize, channels, 0, true);
  MakeAlias(gradientAsCube, gradient, outRowSize, outColSize, channels, 0,
      false);

  for (size_t i = 0; i < outRowSize; ++i)
  {
    size_t rOrigin = std::floor(i / scaleFactors[0]);
    for (size_t j = 0; j < outColSize; ++j)
    {
      size_t cOrigin = std::floor(j / scaleFactors[1]);
      for (size_t k = 0; k < channels; ++k)
      {
        outputAsCube(rOrigin, cOrigin, k) += gradientAsCube(i, j, k);
      }
    }
  }
}

template<typename MatType>
void NearestInterpolationType<MatType>::ComputeOutputDimensions()
{
  if (this->inputDimensions.size() < scaleFactors.size())
  {
    std::ostringstream oss;
    oss << "NearestInterpolation::ComputeOutputDimensions(): input dimensions "
        << "must be at least 2 (received input with "
        << this->inputDimensions.size() << " dimensions)!";
    throw std::runtime_error(oss.str());
  }
  this->outputDimensions = this->inputDimensions;
  for (size_t i = 0; i < scaleFactors.size(); i++)
  {
    this->outputDimensions[i] = std::round(
      (double) this->outputDimensions[i] * scaleFactors[i]);
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
