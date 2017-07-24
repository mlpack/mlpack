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

inline BinaryRBMPolicy::BinaryRBMPolicy(size_t visibleSize, size_t hiddenSize) :
    visibleSize(visibleSize),
    hiddenSize(hiddenSize)
{
  parameter.set_size((visibleSize * hiddenSize) + visibleSize + hiddenSize, 1);
}

// Reset function
inline void BinaryRBMPolicy::Reset()
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
inline double BinaryRBMPolicy::FreeEnergy(arma::mat&& input)
{
  HiddenPreActivation(std::move(input), std::move(preActivation));
  SoftplusFunction::Fn(preActivation, preActivation);
  return  -(arma::accu(preActivation) + arma::dot(input, visibleBias));
}

inline double BinaryRBMPolicy::Evaluate(arma::mat& predictors, size_t i)
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
inline void BinaryRBMPolicy::PositivePhase(arma::mat&& input,
    arma::mat&& gradient)
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
inline void BinaryRBMPolicy::NegativePhase(arma::mat&& negativeSamples,
    arma::mat&& gradient)
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

inline void BinaryRBMPolicy::VisibleMean(arma::mat&& input, arma::mat&& output)
{
  VisiblePreActivation(std::move(input), std::move(output));
  LogisticFunction::Fn(output, output);
}

inline void BinaryRBMPolicy::HiddenMean(arma::mat&& input, arma::mat&& output)
{
  HiddenPreActivation(std::move(input), std::move(output));
  LogisticFunction::Fn(output, output);
}

inline void BinaryRBMPolicy::SampleVisible(arma::mat&& input,
    arma::mat&& output)
{
  VisibleMean(std::move(input), std::move(output));

  for (size_t i = 0; i < output.n_elem; i++)
    output(i) = math::RandBernoulli(output(i));
}

inline void BinaryRBMPolicy::SampleHidden(arma::mat&& input, arma::mat&& output)
{
  HiddenMean(std::move(input), std::move(output));

  for (size_t i = 0; i < output.n_elem; i++)
    output(i) = math::RandBernoulli(output(i));
}

inline void BinaryRBMPolicy::VisiblePreActivation(arma::mat&& input,
    arma::mat&& output)
{
  output = weight.t() * input + visibleBias;
}

inline void BinaryRBMPolicy::HiddenPreActivation(arma::mat&& input,
    arma::mat&& output)
{
  output = weight * input + hiddenBias;
}

template<typename Archive>
void BinaryRBMPolicy::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(visibleSize, "visibleSize");
  ar & data::CreateNVP(hiddenSize, "hiddenSize");
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(weight, "weight");
  ar & data::CreateNVP(visibleBias, "visibleBias");
  ar & data::CreateNVP(hiddenBias, "hiddenBias");
}

} // namespace ann
} // namespace mlpack
#endif // MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_IMPL_HPP
