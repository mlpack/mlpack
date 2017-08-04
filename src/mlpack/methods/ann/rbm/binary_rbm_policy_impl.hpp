/**
 * @file binary_rbm.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_IMPL_HPP
#define MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_IMPL_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>

#include "binary_rbm_policy.hpp"

namespace mlpack {
namespace ann {

template<typename DataType>
inline BinaryRBMPolicy<DataType>
    ::BinaryRBMPolicy(size_t visibleSize, size_t hiddenSize) :
    visibleSize(visibleSize),
    hiddenSize(hiddenSize)
{
  parameter.set_size((visibleSize * hiddenSize) + visibleSize + hiddenSize, 1);
}

// Reset function
template<typename DataType>
inline void BinaryRBMPolicy<DataType>::Reset()
{
  weight = DataType(parameter.memptr(), hiddenSize, visibleSize, false, false);
  hiddenBias = DataType(parameter.memptr() + weight.n_elem,
      hiddenSize, 1, false, false);
  visibleBias = DataType(parameter.memptr() + weight.n_elem +
      hiddenBias.n_elem , visibleSize, 1, false, false);
}

/**
 * Free energy of the spike and slab variable
 * the free energy of the ssRBM is given my
 *
 * @param input the visible layer
 */ 
template<typename DataType>
inline double BinaryRBMPolicy<DataType>
    ::FreeEnergy(DataType&& input)
{
  HiddenPreActivation(std::move(input), std::move(preActivation));
  SoftplusFunction::Fn(preActivation, preActivation);
  return  -(arma::accu(preActivation) + arma::dot(input, visibleBias));
}

template<typename DataType>
inline double BinaryRBMPolicy<DataType>
    ::Evaluate(DataType& predictors, size_t i)
{
  size_t idx = RandInt(0, predictors.n_rows);
  DataType temp = arma::round(predictors.col(i));
  corruptInput = temp;
  corruptInput.row(idx) = 1 - corruptInput.row(idx);
  return std::log(LogisticFunction::Fn(FreeEnergy(std::move(corruptInput)) -
      FreeEnergy(std::move(temp)))) * predictors.n_rows;
}

/**
 * Positive Gradient function. This function calculates the positive
 * phase for the binary rbm gradient calculation
 * 
 * @param input the visible layer type
 */
template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::PositivePhase(DataType&& input, DataType&& gradient)
{
  DataType weightGrad = DataType(gradient.memptr(),
      hiddenSize, visibleSize, false, false);

  DataType hiddenBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  DataType visibleBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem +
      hiddenBiasGrad.n_elem, visibleSize, 1, false, false);

  HiddenMean(std::move(input), std::move(hiddenBiasGrad));
  weightGrad = hiddenBiasGrad * input.t();
  visibleBiasGrad = input;
}

/**
 * Negative Gradient function. This function calculates the negative
 * phase for the binary rbm gradient calculation
 * 
 * @param input the negative samples sampled from gibbs distribution
 */
template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::NegativePhase(DataType&& negativeSamples, DataType&& gradient)
{
  DataType weightGrad = DataType(gradient.memptr(),
      hiddenSize, visibleSize, false, false);

  DataType hiddenBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  DataType visibleBiasGrad = DataType(gradient.memptr() + weightGrad.n_elem +
      hiddenBiasGrad.n_elem, visibleSize, 1, false, false);

  HiddenMean(std::move(negativeSamples), std::move(hiddenBiasGrad));
  weightGrad = hiddenBiasGrad * negativeSamples.t();
  visibleBiasGrad = negativeSamples;
}

template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::VisibleMean(DataType&& input, DataType&& output)
{
  VisiblePreActivation(std::move(input), std::move(output));
  LogisticFunction::Fn(output, output);
}

template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::HiddenMean(DataType&& input, DataType&& output)
{
  HiddenPreActivation(std::move(input), std::move(output));
  LogisticFunction::Fn(output, output);
}

template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::SampleVisible(DataType&& input, DataType&& output)
{
  VisibleMean(std::move(input), std::move(output));

  for (size_t i = 0; i < output.n_elem; i++)
    output(i) = math::RandBernoulli(output(i));
}

template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::SampleHidden(DataType&& input, DataType&& output)
{
  HiddenMean(std::move(input), std::move(output));

  for (size_t i = 0; i < output.n_elem; i++)
    output(i) = math::RandBernoulli(output(i));
}

template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::VisiblePreActivation(DataType&& input, DataType&& output)
{
  output = weight.t() * input + visibleBias;
}

template<typename DataType>
inline void BinaryRBMPolicy<DataType>
    ::HiddenPreActivation(DataType&& input, DataType&& output)
{
  output = weight * input + hiddenBias;
}

template<typename DataType>
template<typename Archive>
void BinaryRBMPolicy<DataType>
    ::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(visibleSize, "visibleSize");
  ar & data::CreateNVP(hiddenSize, "hiddenSize");
}

} // namespace ann
} // namespace mlpack
#endif // MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_IMPL_HPP
