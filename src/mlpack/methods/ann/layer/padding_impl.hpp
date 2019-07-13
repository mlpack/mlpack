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
    const size_t padW,
    const size_t padH) :
    padW(padW),
    padH(padH),
    nRows(0),
    nCols(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Padding<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  nRows = input.n_rows;
  nCols = input.n_cols;
  output = arma::zeros(input.n_rows + padW * 2, input.n_cols + padH * 2);
  output.submat(padW, padH, padW + input.n_rows - 1,
        padH + input.n_cols - 1) = input;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Padding<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */,
    const arma::Mat<eT>&& gy,
    arma::Mat<eT>&& g)
{
  g = gy.submat(padW, padH, padW + nRows - 1,
        padH + nCols - 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Padding<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(padW);
  ar & BOOST_SERIALIZATION_NVP(padH);
}

} // namespace ann
} // namespace mlpack

#endif
