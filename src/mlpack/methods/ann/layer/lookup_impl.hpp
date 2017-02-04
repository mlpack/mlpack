/**
 * @file lookup_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Lookup class a particular convolution, where the width
 * of the convolution is 1.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LOOKUP_IMPL_HPP

// In case it hasn't yet been included.
#include "lookup.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename InputDataType, typename OutputDataType>
Lookup<InputDataType, OutputDataType>::Lookup(
    const size_t inSize,
    const size_t outSize) :
    inSize(inSize),
    outSize(outSize)
{
  weights.set_size(outSize, inSize);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  output = weights.cols(arma::conv_to<arma::uvec>::from(input) - 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */,
    const arma::Mat<eT>&& gy,
    arma::Mat<eT>&& g)
{
  g = gy;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Lookup<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  gradient = arma::zeros<arma::Mat<eT> >(weights.n_rows, weights.n_cols);
  gradient.cols(arma::conv_to<arma::uvec>::from(input) - 1) = error;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Lookup<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(weights, "weights");
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(outSize, "outSize");
}

} // namespace ann
} // namespace mlpack

#endif
