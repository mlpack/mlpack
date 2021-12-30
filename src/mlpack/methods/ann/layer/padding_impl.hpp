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

template<typename InputDataType, typename OutputDataType>
Padding<InputDataType, OutputDataType>::Padding(
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
    nRows(0),
    nCols(0),
    inputHeight(inputWidth),
    inputWidth(inputHeight),
    inSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Padding<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  nRows = input.n_rows;
  nCols = input.n_cols;

  if (inputWidth == 0 || inputHeight == 0)
  {
    output = arma::zeros(nRows + padWLeft + padWRight,
        nCols + padHTop + padHBottom);
    output.submat(padWLeft, padHTop, padWLeft + nRows - 1,
        padHTop + nCols - 1) = input;
  }
  else
  {
    inSize = input.n_elem / (inputWidth * inputHeight * nCols);
    inputTemp = arma::Cube<eT>(const_cast<arma::Mat<eT>&>(input).memptr(),
        inputWidth, inputHeight, inSize * nCols, false, false);
    outputTemp = arma::zeros<arma::Cube<eT>>(inputWidth + padWLeft + padWRight,
        inputHeight + padHTop + padHBottom, inSize * nCols);
    for (size_t i = 0; i < inputTemp.n_slices; ++i)
    {
      outputTemp.slice(i).submat(padWLeft, padHTop, padWLeft + inputWidth - 1,
          padHTop + inputHeight - 1) = inputTemp.slice(i);
    }

    output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem / nCols,
        nCols);
  }

  outputWidth = inputWidth + padWLeft + padWRight;
  outputHeight = inputHeight + padHTop + padHBottom;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Padding<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = gy.submat(padWLeft, padHTop, padWLeft + nRows - 1,
      padHTop + nCols - 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Padding<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(padWLeft));
  ar(CEREAL_NVP(padWRight));
  ar(CEREAL_NVP(padHTop));
  ar(CEREAL_NVP(padHBottom));
  ar(CEREAL_NVP(inputWidth));
  ar(CEREAL_NVP(inputHeight));
}

} // namespace ann
} // namespace mlpack

#endif
