/**
 * @file methods/ann/layer/padding_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the Padding class that adds padding to the incoming
 * data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PADDING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_PADDING_IMPL_HPP

// In case it hasn't yet been included.
#include "padding.hpp"

namespace mlpack {

template<typename MatType>
Padding<MatType>::Padding(
    const size_t padWLeft,
    const size_t padWRight,
    const size_t padHTop,
    const size_t padHBottom,
    const typename MatType::elem_type fillValue) :
    Layer<MatType>(),
    padWLeft(padWLeft),
    padWRight(padWRight),
    padHTop(padHTop),
    padHBottom(padHBottom),
    totalInMaps(0),
    fillValue(fillValue)
{
  // Nothing to do here.
}

template<typename MatType>
Padding<MatType>::Padding(const Padding& other) :
    Layer<MatType>(other),
    padWLeft(other.padWLeft),
    padWRight(other.padWRight),
    padHTop(other.padHTop),
    padHBottom(other.padHBottom),
    fillValue(other.fillValue)
{
  // Nothing to do here.
}

template<typename MatType>
Padding<MatType>::Padding(Padding&& other) :
    Layer<MatType>(std::move(other)),
    padWLeft(std::move(other.padWLeft)),
    padWRight(std::move(other.padWRight)),
    padHTop(std::move(other.padHTop)),
    padHBottom(std::move(other.padHBottom)),
    totalInMaps(std::move(other.totalInMaps)),
    fillValue(std::move(other.fillValue))
{
  // Nothing to do here.
}

template<typename MatType>
Padding<MatType>&
Padding<MatType>::operator=(const Padding& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    padWLeft = other.padWLeft;
    padWRight = other.padWRight;
    padHTop = other.padHTop;
    padHBottom = other.padHBottom;
    totalInMaps = other.totalInMaps;
    fillValue = other.fillValue;
  }

  return *this;
}

template<typename MatType>
Padding<MatType>&
Padding<MatType>::operator=(Padding&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    padWLeft = std::move(other.padWLeft);
    padWRight = std::move(other.padWRight);
    padHTop = std::move(other.padHTop);
    padHBottom = std::move(other.padHBottom);
    totalInMaps = std::move(other.totalInMaps);
    fillValue = std::move(other.fillValue);
  }

  return *this;
}

template<typename MatType>
void Padding<MatType>::Forward(const MatType& input, MatType& output)
{
  // Make an alias of the input and output so that we can deal with the first
  // two dimensions directly.
  CubeType reshapedInput;
  MakeAlias(reshapedInput, input, this->inputDimensions[0],
      this->inputDimensions[1], totalInMaps * input.n_cols, 0, true);
  CubeType reshapedOutput;
  MakeAlias(reshapedOutput, output, this->outputDimensions[0],
      this->outputDimensions[1], totalInMaps * output.n_cols, 0, true);

  // Set the padding parts to 0.
  if (padHTop > 0)
  {
    reshapedOutput.tube(0,
                        0,
                        reshapedOutput.n_rows - 1,
                        padHTop - 1).fill(fillValue);
  }

  if (padWLeft > 0)
  {
    reshapedOutput.tube(0,
                        padHTop,
                        padWLeft - 1,
                        padHTop + this->inputDimensions[1] - 1).fill(fillValue);
  }

  if (padHBottom > 0)
  {
    reshapedOutput.tube(0,
                        padHTop + this->inputDimensions[1],
                        reshapedOutput.n_rows - 1,
                        reshapedOutput.n_cols - 1).fill(fillValue);
  }

  if (padWRight > 0)
  {
    reshapedOutput.tube(padWLeft + this->inputDimensions[0],
                        padHTop,
                        reshapedOutput.n_rows - 1,
                        padHTop + this->inputDimensions[1] - 1).fill(fillValue);
  }

  // Copy the input matrix.
  reshapedOutput.tube(padWLeft,
                      padHTop,
                      padWLeft + this->inputDimensions[0] - 1,
                      padHTop + this->inputDimensions[1] - 1) = reshapedInput;
}

template<typename MatType>
void Padding<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  // Reshape g and gy so that extracting the un-padded input is easier to
  // understand.
  CubeType reshapedGy;
  MakeAlias(reshapedGy, gy, this->outputDimensions[0],
      this->outputDimensions[1], totalInMaps * gy.n_cols, 0, true);
  CubeType reshapedG;
  MakeAlias(reshapedG, g, this->inputDimensions[0],
      this->inputDimensions[1], totalInMaps * g.n_cols, 0, true);

  reshapedG = reshapedGy.tube(padWLeft,
                              padHTop,
                              padWLeft + this->inputDimensions[0] - 1,
                              padHTop + this->inputDimensions[1] - 1);
}

template<typename MatType>
void Padding<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;

  this->outputDimensions[0] += padWLeft + padWRight;
  this->outputDimensions[1] += padHTop + padHBottom;

  // Higher dimensions remain unchanged.  But, we will cache the product of
  // these higher dimensions.
  totalInMaps = 1;
  for (size_t i = 2; i < this->inputDimensions.size(); ++i)
    totalInMaps *= this->inputDimensions[i];
}

template<typename MatType>
template<typename Archive>
void Padding<MatType>::serialize(Archive& ar, const uint32_t version)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(padHBottom));
  ar(CEREAL_NVP(totalInMaps));

  if (version == 0)
  {
    fillValue = 0;
  }
  else
  {
    ar(CEREAL_NVP(fillValue));
  }
}

} // namespace mlpack

#endif
