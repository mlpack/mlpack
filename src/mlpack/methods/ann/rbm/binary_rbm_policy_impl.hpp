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

template <typename InputDataType, typename OutputDataType>
inline BinaryRBMPolicy<InputDataType, OutputDataType>
    ::BinaryRBMPolicy(size_t visibleSize, size_t hiddenSize) :
    visibleSize(visibleSize),
    hiddenSize(hiddenSize)
{
  parameter.set_size((visibleSize * hiddenSize) + visibleSize + hiddenSize, 1);
}

// Reset function
template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>::Reset()
{
  weight = arma::mat(parameter.memptr(), hiddenSize, visibleSize, false, false);
  hiddenBias = arma::mat(parameter.memptr() + weight.n_elem,
      hiddenSize, 1, false, false);
  visibleBias = arma::mat(parameter.memptr() + weight.n_elem +
      hiddenBias.n_elem , visibleSize, 1, false, false);
}

/**
 * Free energy of the spike and slab variable
 * the free energy of the ssRBM is given my
 *
 * @param input the visible layer
 */ 
template <typename InputDataType, typename OutputDataType>
inline double BinaryRBMPolicy<InputDataType, OutputDataType>
    ::FreeEnergy(InputDataType&& input)
{
  HiddenPreActivation(std::move(input), std::move(preActivation));
  SoftplusFunction::Fn(preActivation, preActivation);
  return  -(arma::accu(preActivation) + arma::dot(input, visibleBias));
}

template <typename InputDataType, typename OutputDataType>
inline double BinaryRBMPolicy<InputDataType, OutputDataType>
    ::Evaluate(InputDataType& predictors, size_t i)
{
  size_t idx = RandInt(0, predictors.n_rows);
  arma::mat temp = arma::round(predictors.col(i));
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
template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::PositivePhase(InputDataType&& input, OutputDataType&& gradient)
{
  arma::mat weightGrad = arma::mat(gradient.memptr(),
      hiddenSize, visibleSize, false, false);

  arma::mat hiddenBiasGrad = arma::mat(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  arma::mat visibleBiasGrad = arma::mat(gradient.memptr() + weightGrad.n_elem +
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
template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::NegativePhase(InputDataType&& negativeSamples, OutputDataType&& gradient)
{
  arma::mat weightGrad = arma::mat(gradient.memptr(),
      hiddenSize, visibleSize, false, false);

  arma::mat hiddenBiasGrad = arma::mat(gradient.memptr() + weightGrad.n_elem,
      hiddenSize, 1, false, false);

  arma::mat visibleBiasGrad = arma::mat(gradient.memptr() + weightGrad.n_elem +
      hiddenBiasGrad.n_elem, visibleSize, 1, false, false);

  HiddenMean(std::move(negativeSamples), std::move(hiddenBiasGrad));
  weightGrad = hiddenBiasGrad * negativeSamples.t();
  visibleBiasGrad = negativeSamples;
}

template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::VisibleMean(InputDataType&& input, OutputDataType&& output)
{
  VisiblePreActivation(std::move(input), std::move(output));
  LogisticFunction::Fn(output, output);
}

template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::HiddenMean(InputDataType&& input, OutputDataType&& output)
{
  HiddenPreActivation(std::move(input), std::move(output));
  LogisticFunction::Fn(output, output);
}

template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::SampleVisible(InputDataType&& input, OutputDataType&& output)
{
  VisibleMean(std::move(input), std::move(output));

  for (size_t i = 0; i < output.n_elem; i++)
    output(i) = math::RandBernoulli(output(i));
}

template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::SampleHidden(InputDataType&& input, OutputDataType&& output)
{
  HiddenMean(std::move(input), std::move(output));

  for (size_t i = 0; i < output.n_elem; i++)
    output(i) = math::RandBernoulli(output(i));
}

template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::VisiblePreActivation(InputDataType&& input, OutputDataType&& output)
{
  output = weight.t() * input + visibleBias;
}

template <typename InputDataType, typename OutputDataType>
inline void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::HiddenPreActivation(InputDataType&& input, OutputDataType&& output)
{
  output = weight * input + hiddenBias;
}

template <typename InputDataType, typename OutputDataType>
template<typename Archive>
void BinaryRBMPolicy<InputDataType, OutputDataType>
    ::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(visibleSize, "visibleSize");
  ar & data::CreateNVP(hiddenSize, "hiddenSize");
}

} // namespace ann
} // namespace mlpack
#endif // MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_IMPL_HPP
