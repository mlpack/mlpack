/**
 * @file add_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Add class that applies a bias term to the incoming
 * data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_IMPL_HPP

// In case it hasn't yet been included.
#include "add.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Add<InputDataType, OutputDataType>::Add(const size_t outSize) :
    outSize(outSize)
{
  weights.set_size(outSize, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Add<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  output = input + weights;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Add<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */,
    const arma::Mat<eT>&& gy,
    arma::Mat<eT>&& g)
{
  g = gy;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Add<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  gradient = error;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Add<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(weights, "weights");
}

} // namespace ann
} // namespace mlpack

#endif
