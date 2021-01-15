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
    const size_t padHBottom) :
    padWLeft(padWLeft),
    padWRight(padWRight),
    padHTop(padHTop),
    padHBottom(padHBottom),
    nRows(0),
    nCols(0)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void PaddingType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  nRows = input.n_rows;
  nCols = input.n_cols;
  output = arma::zeros(nRows + padWLeft + padWRight,
      nCols + padHTop + padHBottom);
  output.submat(padWLeft, padHTop, padWLeft + nRows - 1,
      padHTop + nCols - 1) = input;
}

template<typename InputType, typename OutputType>
void PaddingType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  g = gy.submat(padWLeft, padHTop, padWLeft + nRows - 1,
      padHTop + nCols - 1);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void PaddingType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(padHBottom));
}

} // namespace ann
} // namespace mlpack

#endif
