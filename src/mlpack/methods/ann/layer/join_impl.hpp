/**
 * @file methods/ann/layer/join_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Join module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_JOIN_IMPL_HPP

// In case it hasn't yet been included.
#include "join.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Join<InputDataType, OutputDataType>::Join() :
    inSizeRows(0),
    inSizeCols(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Join<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  inSizeRows = input.n_rows;
  inSizeCols = input.n_cols;
  output = arma::vectorise(input);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Join<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = arma::mat(((arma::Mat<eT>&) gy).memptr(), inSizeRows, inSizeCols, false,
      false);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Join<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inSizeRows);
  ar & BOOST_SERIALIZATION_NVP(inSizeCols);
}

} // namespace ann
} // namespace mlpack

#endif
