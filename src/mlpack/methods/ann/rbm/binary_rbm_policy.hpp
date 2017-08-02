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
template <typename InputDataType = arma::mat,
          typename OutputDataType = arma::mat>
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
  double FreeEnergy(InputDataType&& input);

  double Evaluate(InputDataType& predictors, size_t i);

  /**
   * Positive Gradient function. This function calculates the positive
   * phase for the binary rbm gradient calculation
   * 
   * @param input the visible layer type
   */
  void PositivePhase(InputDataType&& input, OutputDataType&& gradient);

  /**
   * Negative Gradient function. This function calculates the negative
   * phase for the binary rbm gradient calculation
   * 
   * @param input the negative samples sampled from gibbs distribution
   */
  void NegativePhase(InputDataType&& negativeSamples,
      OutputDataType&& gradient);

  /**
   * Visible mean function calcultes the forward pass for
   * the visible layer.
   *
   * @param input hidden neurons
   * @param output visible neuron activations
   */
  void VisibleMean(InputDataType && input, OutputDataType&& output);

  /**
   * Hidden mean function calcultes the forward pass for
   * the hidden layer.
   *
   * @param input visible neurons
   * @param output hidden neuron activations
   */
  void HiddenMean(InputDataType&& input, OutputDataType&& output);

  /**
   * SampleVisible function samples the visible 
   * layer using bernoulli function.
   *
   * @param input hidden neurons
   * @param output sampled visible neurons
   */
  void SampleVisible(InputDataType&& input, OutputDataType&& output);

  /**
   * SampleHidden function samples the hidden 
   * layer using bernoulli function.
   *
   * @param input visible neurons
   * @param output sampled hidden neurons
   */
  void SampleHidden(InputDataType&& input, OutputDataType&& output);

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Return the initial point for the optimization.
  const InputDataType& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  InputDataType& Parameters() { return parameter; }
  //! Return the weights of the network
  const OutputDataType& Weight() const { return weight; }
  //! Modify the weight of the network
  OutputDataType& Weight() { return weight; }

  //! Return the visible bias of the network
  const InputDataType& VisibleBias() const { return visibleBias; }
  //! Modify the visible bias of the network
  InputDataType& VisibleBias() { return visibleBias; }

  //! Return the hidden bias of the network
  const InputDataType& HiddenBias() const { return hiddenBias; }
  //! Modify the  hidden bias of the network
  InputDataType& HiddenBias() { return hiddenBias; }

  //! Get the visible size
  size_t const& VisibleSize() const { return visibleSize; }
  //! Get the hidden size
  size_t const& HiddenSize() const { return hiddenSize; }

 private:
  /**
   * VisiblePreAction function calculates the pre activation
   * values given the hidden input units.
   *
   * @param input hidden unit neuron
   * @param ouput visible unit pre-activation values
   */
  void VisiblePreActivation(InputDataType&& input, OutputDataType&& output);
  /**
   * HiddenPreActivation function calculates the pre activation
   * values given the hidden input units.
   *
   * @param input visible unit neuron
   * @param ouput hidden unit pre-activation values
   */
  void HiddenPreActivation(InputDataType&& input, OutputDataType&& output);

 private:
  //! Locally stored number of visible neurons
  size_t visibleSize;
  //! Locally stored number of hidden neurons
  size_t hiddenSize;
  //! Locally stored  Parameters of the network
  InputDataType parameter;
  //! Locally stored weight of the network
  InputDataType weight;
  //! Locally stored biases of the visible layer
  OutputDataType visibleBias;
  //! Locally stored biases of hidden layer
  OutputDataType hiddenBias;
  //! Locally-stored output of the preActivation function used in FreeEnergy
  InputDataType preActivation;
  //! Locally-stored corrupInput used for Pseudo-Likelihood
  InputDataType corruptInput;
};
} // namespace ann
} // namespace mlpack

#include "binary_rbm_policy_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_HPP
