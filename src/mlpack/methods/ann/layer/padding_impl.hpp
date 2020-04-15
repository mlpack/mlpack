/**
 * @file padding_impl.hpp
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
    const std::string& paddingType,
    const double fillNum) :
    padWLeft(padWLeft),
    padWRight(padWRight),
    padHTop(padHTop),
    padHBottom(padHBottom),
    fillNum(fillNum),
    nRows(0),
    nCols(0)
{
  std::transform(paddingType.begin(), paddingType.end(), paddingTypeLow.begin(),
      [](unsigned char c){ return std::tolower(c); });
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Padding<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  nRows = input.n_rows;
  nCols = input.n_cols;
  output = arma::zeros(nRows + padWLeft + padWRight,
      nCols + padHTop + padHBottom);
  if (paddingTypeLow == "constant")
  {
    output.fill(fillNum);
  }
  else if (paddingTypeLow == "reflective")
  {
    // Perform reflective padding.
  }
  output.submat(padWLeft, padHTop, padWLeft + nRows - 1,
      padHTop + nCols - 1) = input;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Padding<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */,
    const arma::Mat<eT>&& gy,
    arma::Mat<eT>&& g)
{
  g = gy.submat(padWLeft, padHTop, padWLeft + nRows - 1,
      padHTop + nCols - 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Padding<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(padWLeft);
  ar & BOOST_SERIALIZATION_NVP(padWRight);
  ar & BOOST_SERIALIZATION_NVP(padHTop);
  ar & BOOST_SERIALIZATION_NVP(padHBottom);
  ar & BOOST_SERIALIZATION_NVP(fillNum);
}

} // namespace ann
} // namespace mlpack

#endif
