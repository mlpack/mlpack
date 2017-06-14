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

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

namespace mlpack{
namespace ann{
/**
 * The BinaryRBMPolicy class
 *
 * @tparam DataType the type of matrix to be used.
 */
template <typename DataType = arma::mat>
class BinaryRBMPolicy
{
 public:
  typedef typename DataType::elem_type ElemType;
  /**
   * Intialise the visible and hidden layer of the network.
   * @param visibelSize Number of visible neurons.
   * @param hiddenSize Number of hidden neurons.
   */
  BinaryRBMPolicy(const size_t visibleSize, const size_t hiddenSize);

  // Reset function
  void Reset();

  /**
   * This function calculates the Free Energy of the binary RBM.
   * The free energy is given by
   * $-b^Tv - \sum_{i=1}^M log(1 + e^{c_j+v^TW_j})$.
   *
   * @param input The visible neurons.
   */ 
  ElemType FreeEnergy(DataType&& input);

  /**
   * Evaluate function computes the perfomance of the RBM at the given.
   * input in the case persistent = true
   *
   * @param predictors Training data of the network.
   * @param i The idx of the current input.
   */ 
  ElemType Evaluate(DataType& predictors, size_t i);

  /**
   * Calculates the Gradient of the RBM network on the 
   * visible input obtained from the training data.
   * 
   * @param input The visible layer type.
   * @param gradient Stores the gradient of the rbm network.
   */
  void PositivePhase(DataType&& input, DataType&& gradient);

  /**
   * Calculate the Gradient of the RBM network on the sampled
   * visible input obtained from gibbs sampling.
   * 
   * @param input The negative samples sampled from gibbs distribution.
   * @param gradient Stores the gradient of the rbm network.
   */
  void NegativePhase(DataType&& negativeSamples, DataType&& gradient);

  /**
   * The function calculates the mean for the visible layer.
   *
   * @param input Hidden neurons from the hidden layer of the network.
   * @param output Visible neuron activations.
   */
  void VisibleMean(DataType&& input, DataType&& output);

  /**
   * The function calcultes the mean for the hidden layer.
   *
   * @param input Visible neurons.
   * @param output Hidden neuron activations.
   */
  void HiddenMean(DataType&& input, DataType&& output);

  /**
   * SampleVisible function samples the visible layer using bernoulli function.
   *
   * @param input Hidden neurons.
   * @param output Sampled visible neurons.
   */
  void SampleVisible(DataType&& input, DataType&& output);

  /**
   * SampleHidden function samples the hidden layer using bernoulli function.
   *
   * @param input Visible neurons.
   * @param output Sampled hidden neurons.
   */
  void SampleHidden(DataType&& input, DataType&& output);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  //! Return the parameters of the network.
  const DataType& Parameters() const { return parameter; }
  //! Modify the parameters of the network.
  DataType& Parameters() { return parameter; }
  //! Return the weights of the network.
  const DataType& Weight() const { return weight; }
  //! Modify the weight of the network.
  DataType& Weight() { return weight; }

  //! Return the visible bias of the network.
  const DataType& VisibleBias() const { return visibleBias; }
  //! Modify the visible bias of the network.
  DataType& VisibleBias() { return visibleBias; }

  //! Return the hidden bias of the network.
  const DataType& HiddenBias() const { return hiddenBias; }
  //! Modify the  hidden bias of the network.
  DataType& HiddenBias() { return hiddenBias; }

  //! Get the visible size.
  size_t const& VisibleSize() const { return visibleSize; }
  //! Get the hidden size.
  size_t const& HiddenSize() const { return hiddenSize; }

 private:
  /**
   * VisiblePreAction function calculates the pre activation
   * values given the hidden input units.
   *
   * @param input Hidden neurons.
   * @param ouput Visible unit pre-activation values.
   */
  void VisiblePreActivation(DataType&& input, DataType&& output);
  /**
   * HiddenPreActivation function calculates the pre activation
   * values given the visible input units.
   *
   * @param input Visible unit neuron.
   * @param ouput Hidden unit pre-activation values.
   */
  void HiddenPreActivation(DataType&& input, DataType&& output);

 private:
  //! Locally stored number of visible neurons.
  size_t visibleSize;
  //! Locally stored number of hidden neurons
  size_t hiddenSize;
  //! Locally stored  Parameters of the network.
  DataType parameter;
  //! Locally stored weight of the network.
  DataType weight;
  //! Locally stored biases of the visible layer.
  DataType visibleBias;
  //! Locally stored biases of hidden layer.
  DataType hiddenBias;
  //! Locally-stored output of the preActivation function used in FreeEnergy.
  DataType preActivation;
  //! Locally-stored corrupInput used for Pseudo-Likelihood.
  DataType corruptInput;
};
} // namespace ann
} // namespace mlpack

#include "binary_rbm_policy_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_BINARY_RBM_POLICY_HPP
