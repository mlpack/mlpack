/**
 * @file binary_rbm_impl.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BINARY_RBM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_BINARY_RBM_IMPL_HPP
// In case it hasn't yet been included.
#include "binary_rbm.hpp"

#include <mlpack/core/math/random.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
VisibleLayer<InputDataType, OutputDataType>::VisibleLayer(
  const size_t inSize,
	const size_t outSize):
  inSize(inSize), 
  outSize(outSize)
{
  weights.set_size(outSize * inSize + inSize, 1);
}

template<typename InputDataType, typename OutputDataType>
void VisibleLayer<InputDataType, OutputDataType>::Reset()
{
  
  weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
  ownBias = arma::mat(weights.memptr() + weight.n_elem, inSize, false, false);
  otherBias = arma::mat(weight.memptr() + weight.n_elem + inSize, outSize, false, false);

}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void VisibleLayer<InputDataType, OutputDataType>::Forward(
  arma::Mat<eT>&& input, 
  arma::Mat<eT>&& output)
{ 
  if(Parameters().empty())
    Reset();
  else
    LogisticFunction::Fn(weight * input + otherBias, output);
}
template<typename InputDataType, typename OutputDataType>
template<typename eT>
void VisibleLayer<InputDataType, OutputDataType>::Sample(arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
    Forward(std::move(input), std::move(output));
    for(size_t i = 0; i < output.n_elem; i++)
      if (mlpack::math::Random() > output(i))
        output(i) = 1;
      else
        output(i) = 0;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
double VisibleLayer<InputDataType, OutputDataType>::FreeEnergy(arma::Mat<eT>&& input)
{
  arma::Mat<eT> output(input.n_elem);
  for(size_t i = 0; i < weight.n_cols; i++)
    output(i) = SoftplusFunction::Fn(arma::accu(weight.cols(i) % input));
  return arma::dot(input, ownBias) + arma::accu(output);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void VisibleLayer<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(weights, "weights");
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(outSize, "outSize");
}
} // ann
} // mlpack
#endif