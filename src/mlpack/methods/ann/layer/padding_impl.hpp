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
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
PaddingType<InputType, OutputType>::PaddingType(
    const size_t padWLeft,
    const size_t padWRight,
    const size_t padHTop,
    const size_t padHBottom,
    const size_t inputWidth,
    const size_t inputHeight) :
    padWLeft(padWLeft),
    padWRight(padWRight),
    padHTop(padHTop),
    padHBottom(padHBottom),
    totalInMaps(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void PaddingType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  // Make an alias of the input and output so that we can deal with the first
  // two dimensions directly.
  arma::Cube<typename InputType::elem_type> reshapedInput(
      (typename InputType::elem_type*) input.memptr(),
      this->inputDimensions[0], this->inputDimensions[1], totalInMaps *
      input.n_cols, false, true);
  arma::Cube<typename OutputType::elem_type> reshapedOutput(output.memptr(),
      this->outputDimensions[0], this->outputDimensions[1], totalInMaps *
      output.n_cols, false, true);

  // Set the padding parts to 0.
  if (padWLeft > 0)
  {
    reshapedOutput.tube(0,
                        0,
                        padWLeft - 1,
                        reshapedOutput.n_cols - 1).zeros();
  }

  if (padHTop > 0)
  {
    reshapedOutput.tube(padWLeft,
                        0,
                        padWLeft + this->inputDimensions[0] - 1,
                        padHTop - 1).zeros();
  }

  if (padWRight > 0)
  {
    reshapedOutput.tube(padWLeft + this->inputDimensions[0],
                        padHTop + this->inputDimensions[1],
                        reshapedOutput.n_rows - 1,
                        reshapedOutput.n_cols - 1).zeros();
  }

  if (padHBottom > 0)
  {
    reshapedOutput.tube(padWLeft,
                        0,
                        padWLeft + this->inputDimensions[0] - 1,
                        reshapedOutput.n_cols - 1).zeros();
  }

  // Copy the input matrix.
  reshapedOutput.tube(padWLeft,
                      padHTop,
                      padWLeft + this->inputDimensions[0] - 1,
                      padHTop + this->inputDimensions[1] - 1) = reshapedInput;
}

template<typename InputType, typename OutputType>
void PaddingType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  // Reshape g and gy so that extracting the un-padded input is easier to
  // understand.
  arma::Cube<typename OutputType::elem_type> reshapedGy(
      (typename OutputType::elem_type*) gy.memptr(), this->outputDimensions[0],
      this->outputDimensions[1], totalInMaps * gy.n_cols, false, true);
  arma::Cube<typename OutputType::elem_type> reshapedG(g.memptr(),
      this->inputDimensions[0], this->inputDimensions[1], totalInMaps *
      g.n_cols, false, true);

  reshapedG = reshapedGy.tube(padWLeft,
                              padHTop,
                              padWLeft + this->inputDimensions[0] - 1,
                              padHTop + this->inputDimensions[1] - 1);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void PaddingType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(padHBottom));
  ar(CEREAL_NVP(totalInMaps));
}

} // namespace ann
} // namespace mlpack

#endif
