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
PaddingType<MatType>::PaddingType(
    const size_t padWLeft,
    const size_t padWRight,
    const size_t padHTop,
    const size_t padHBottom) :
    Layer<MatType>(),
    padWLeft(padWLeft),
    padWRight(padWRight),
    padHTop(padHTop),
    padHBottom(padHBottom),
    totalInMaps(0)
{
  // Nothing to do here.
}

template<typename MatType>
PaddingType<MatType>::PaddingType(const PaddingType& other) :
    Layer<MatType>(other),
    padWLeft(other.padWLeft),
    padWRight(other.padWRight),
    padHTop(other.padHTop),
    padHBottom(other.padHBottom),
    totalInMaps(other.totalInMaps)
{
  // Nothing to do here.
}

template<typename MatType>
PaddingType<MatType>::PaddingType(PaddingType&& other) :
    Layer<MatType>(std::move(other)),
    padWLeft(std::move(other.padWLeft)),
    padWRight(std::move(other.padWRight)),
    padHTop(std::move(other.padHTop)),
    padHBottom(std::move(other.padHBottom)),
    totalInMaps(std::move(other.totalInMaps))
{
  // Nothing to do here.
}

template<typename MatType>
PaddingType<MatType>&
PaddingType<MatType>::operator=(const PaddingType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    padWLeft = other.padWLeft;
    padWRight = other.padWRight;
    padHTop = other.padHTop;
    padHBottom = other.padHBottom;
    totalInMaps = other.totalInMaps;
  }

  return *this;
}

template<typename MatType>
PaddingType<MatType>&
PaddingType<MatType>::operator=(PaddingType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    padWLeft = std::move(other.padWLeft);
    padWRight = std::move(other.padWRight);
    padHTop = std::move(other.padHTop);
    padHBottom = std::move(other.padHBottom);
    totalInMaps = std::move(other.totalInMaps);
  }

  return *this;
}

template<typename MatType>
void PaddingType<MatType>::Forward(const MatType& input, MatType& output)
{
  // Make an alias of the input and output so that we can deal with the first
  // two dimensions directly.
  arma::Cube<typename MatType::elem_type> reshapedInput(
      (typename MatType::elem_type*) input.memptr(),
      this->inputDimensions[0], this->inputDimensions[1], totalInMaps *
      input.n_cols, false, true);
  arma::Cube<typename MatType::elem_type> reshapedOutput(output.memptr(),
      this->outputDimensions[0], this->outputDimensions[1], totalInMaps *
      output.n_cols, false, true);

  // Set the padding parts to 0.
  if (padHTop > 0)
  {
    reshapedOutput.tube(0,
                        0,
                        reshapedOutput.n_rows - 1,
                        padHTop - 1).zeros();
  }

  if (padWLeft > 0)
  {
    reshapedOutput.tube(0,
                        padHTop,
                        padWLeft - 1,
                        padHTop + this->inputDimensions[1] - 1).zeros();
  }

  if (padHBottom > 0)
  {
    reshapedOutput.tube(0,
                        padHTop + this->inputDimensions[1],
                        reshapedOutput.n_rows - 1,
                        reshapedOutput.n_cols - 1).zeros();
  }

  if (padWRight > 0)
  {
    reshapedOutput.tube(padWLeft + this->inputDimensions[0],
                        padHTop,
                        reshapedOutput.n_rows - 1,
                        padHTop + this->inputDimensions[1] - 1).zeros();
  }

  // Copy the input matrix.
  reshapedOutput.tube(padWLeft,
                      padHTop,
                      padWLeft + this->inputDimensions[0] - 1,
                      padHTop + this->inputDimensions[1] - 1) = reshapedInput;
}

template<typename MatType>
void PaddingType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  // Reshape g and gy so that extracting the un-padded input is easier to
  // understand.
  arma::Cube<typename MatType::elem_type> reshapedGy(
      (typename MatType::elem_type*) gy.memptr(), this->outputDimensions[0],
      this->outputDimensions[1], totalInMaps * gy.n_cols, false, true);
  arma::Cube<typename MatType::elem_type> reshapedG(g.memptr(),
      this->inputDimensions[0], this->inputDimensions[1], totalInMaps *
      g.n_cols, false, true);

  reshapedG = reshapedGy.tube(padWLeft,
                              padHTop,
                              padWLeft + this->inputDimensions[0] - 1,
                              padHTop + this->inputDimensions[1] - 1);
}

template<typename MatType>
void PaddingType<MatType>::ComputeOutputDimensions()
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
void PaddingType<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(padHBottom));
  ar(CEREAL_NVP(totalInMaps));
}

} // namespace mlpack

#endif
