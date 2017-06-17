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
BinaryLayer<InputDataType, OutputDataType>::BinaryLayer(
  const size_t inSize,
	const size_t outSize,
  const bool typeVisible):
  inSize(inSize), 
  outSize(outSize),
  typeVisible(typeVisible)
{
  weights.set_size(outSize * inSize + inSize + outSize, 1);
}

template<typename InputDataType, typename OutputDataType>
void BinaryLayer<InputDataType, OutputDataType>::Reset()
{
  if(typeVisible)
  {
    weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
    ownBias = arma::mat(weights.memptr() + weight.n_elem, inSize, 1, false, false);
    otherBias = arma::mat(weights.memptr() + weight.n_elem + inSize, outSize, 1, false, false);
  }
  else
  {
    weight = arma::mat(weights.memptr(), inSize, outSize, false, false);
    ownBias = arma::mat(weights.memptr() + weight.n_elem, inSize, 1, false, false);
    otherBias = arma::mat(weights.memptr() + weight.n_elem + inSize, outSize, 1, false, false);
  }
}

template<typename InputDataType, typename OutputDataType>
void BinaryLayer<InputDataType, OutputDataType>::Forward(
  InputDataType&& input, 
  OutputDataType&& output)
{ 
  if(Parameters().empty())
    Reset();
  else
      LogisticFunction::Fn(weight * input + otherBias, output);
}

template<typename InputDataType, typename OutputDataType>
void BinaryLayer<InputDataType, OutputDataType>::Sample(InputDataType&& input, OutputDataType&& output)
{
    Forward(std::move(input), std::move(output));
    math::BinomialRandom<>(std::move(input), std::move(output));
}

template<typename InputDataType, typename OutputDataType>
double BinaryLayer<InputDataType, OutputDataType>::FreeEnergy(const InputDataType&& input)
{
  OutputDataType output;
  output = SoftplusFunction::Fn(arma::accu(weight.t() * input));
  return arma::dot(input, ownBias) + arma::accu(output);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BinaryLayer<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(weights, "weights");
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(outSize, "outSize");
}
} // ann
} // mlpack
#endif