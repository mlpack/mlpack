/**
 * @file weight_norm_impl.hpp
 * @author Toshal Agrawal
 *
 * Implementation of the Weight Normalization Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_IMPL_HPP

// In case it is not included.
#include "weight_norm.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
WeightNorm<InputDataType, OutputDataType, CustomLayers...>::WeightNorm() :
    model(false),
    scalarParameter(1)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
WeightNorm<InputDataType, OutputDataType, CustomLayers...>::~WeightNorm()
{
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Reset()
{
  // It will call the reset function of weight norm layer.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  // It will call the Forward function of the wrapped layer.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Backward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  // It will directly call the Backward function of the wrapped layer.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename eT>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::Gradient(
    const arma::Mat<eT>&& /* input */,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  // First it will calculate the gradients of the wrapped layer.
  // Then it will calculate gradients of the vector parameter v and scalar
  // parameter g.
}

template<typename InputDataType, typename OutputDataType,
         typename... CustomLayers>
template<typename Archive>
void WeightNorm<InputDataType, OutputDataType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(network);
  ar & BOOST_SERIALIZATION_NVP(model);
  ar & BOOST_SERIALIZATION_NVP(scalarParameter);
  ar & BOOST_SERIALIZATION_NVP(vectorParameter);
}

} // namespace ann
} // namespace mlpack

#endif
