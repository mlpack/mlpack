/**
 * @file binary_rbm_impl.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RBM_BINARY_LAYER_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RBM_BINARY_LAYER_IMPL_HPP
// In case it hasn't yet been included.
#include "binary_layer.hpp"

#include <mlpack/core/math/random.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

using namespace mlpack;
using namespace mlpack::ann;

namespace mlpack {
namespace rbm { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
BinaryLayer<InputDataType, OutputDataType>::BinaryLayer(
    size_t inSize,
    size_t outSize,
    bool typeVisible):
    inSize(inSize),
    outSize(outSize),
    typeVisible(typeVisible)
{
  weights.set_size((outSize * inSize) + inSize + outSize, 1);
}

template<typename InputDataType, typename OutputDataType>
void BinaryLayer<InputDataType, OutputDataType>::Reset()
{
  if (typeVisible)
  {
    weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
    ownBias = arma::mat(weights.memptr() + weight.n_elem + outSize,
        inSize, 1, false, false);
    otherBias = arma::mat(weights.memptr() + weight.n_elem , outSize,
        1, false, false);
  }
  else
  {
    weight = arma::mat(weights.memptr(), inSize, outSize, false, false);
    ownBias = arma::mat(weights.memptr() + weight.n_elem, inSize, 1,
        false, false);
    otherBias = arma::mat(weights.memptr() + weight.n_elem + inSize, outSize, 1,
        false, false);
  }
}

template<typename InputDataType, typename OutputDataType>
void BinaryLayer<InputDataType, OutputDataType>::Forward(
    const InputDataType&& input, OutputDataType&& output)
{
    ForwardPreActivation(std::move(input), std::move(output));
    LogisticFunction::Fn(output, output);
}

template<typename InputDataType, typename OutputDataType>
void BinaryLayer<InputDataType, OutputDataType>::ForwardPreActivation(
    const InputDataType&& input, OutputDataType&& output)
{
    if (Parameters().empty())
      Reset();

    if (typeVisible)
      output = weight * input + otherBias;
    else
      output = weight.t() * input + otherBias;
}

template<typename InputDataType, typename OutputDataType>
void BinaryLayer<InputDataType, OutputDataType>::Sample(InputDataType&& input,
    OutputDataType&& output)
{
  Forward(std::move(input), std::move(output));
  for (size_t i = 0; i < output.n_elem; i++)
    output(i) = math::RandBernoulli(output(i));
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BinaryLayer<InputDataType, OutputDataType>::Serialize(
  Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(weights, "weights");
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(outSize, "outSize");
  ar & data::CreateNVP(typeVisible, "typeVisible");
}
} // namespace rbm
} // namespace mlpack
#endif
