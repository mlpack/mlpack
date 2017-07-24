/**
 * @file binary_rbm_policy.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_HPP
#define MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

namespace mlpack{
namespace ann{

class BinaryRBMPolicy
{
 public:
  // Intialise the visible and hiddenl layer of the network
  BinaryRBMPolicy(size_t visibleSize, size_t hiddenSize);

  // Reset function
  void Reset();

  /**
   * Free energy of the spike and slab variable
   * the free energy of the ssRBM is given my
   *
   * @param input the visible layer
   */ 
  double FreeEnergy(arma::mat&& input);

  double Evaluate(arma::mat& predictors, size_t i);

  /**
   * Positive Gradient function. This function calculates the positive
   * phase for the binary rbm gradient calculation
   * 
   * @param input the visible layer type
   */
  void PositivePhase(arma::mat&& input, arma::mat&& gradient);

  /**
   * Negative Gradient function. This function calculates the negative
   * phase for the binary rbm gradient calculation
   * 
   * @param input the negative samples sampled from gibbs distribution
   */
  void NegativePhase(arma::mat&& negativeSamples, arma::mat&& gradient);

  void VisibleMean(arma::mat&& input, arma::mat&& output);

  void HiddenMean(arma::mat&& input, arma::mat&& output);

  void SampleVisible(arma::mat&& input, arma::mat&& output);

  void SampleHidden(arma::mat&& input, arma::mat&& output);

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  const arma::mat& Weight() const { return weight; }
  arma::mat& Weight() { return weight; }

  const arma::mat& VisibleBias() const { return visibleBias; }
  arma::mat& VisibleBias() { return visibleBias; }

  const arma::mat& HiddenBias() const { return hiddenBias; }
  arma::mat& HiddenBias() { return hiddenBias; }

 private:
  void VisiblePreActivation(arma::mat&& input, arma::mat&& output);
  void HiddenPreActivation(arma::mat&& input, arma::mat&& output);

 private:
  size_t visibleSize;

  size_t hiddenSize;

  // Parameter weights of the network
  arma::mat parameter;

  arma::mat weight;

  arma::mat visibleBias;

  arma::mat hiddenBias;

  //! Locally-stored output of the preActivation function used in FreeEnergy
  arma::mat preActivation;
  //! Locally-stored corrupInput used for Pseudo-Likelihood
  arma::mat corruptInput;
};
} // namespace ann
} // namespace mlpack

#include "binary_rbm_policy_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_HPP
