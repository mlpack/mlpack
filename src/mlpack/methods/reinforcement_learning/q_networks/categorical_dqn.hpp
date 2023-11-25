/**
 * @file methods/reinforcement_learning/q_networks/categorical_dqn.hpp
 * @author Nishant Kumar
 *
 * This file contains the implementation of the categorical deep q network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_CATEGORICAL_DQN_HPP
#define MLPACK_METHODS_RL_CATEGORICAL_DQN_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/empty_loss.hpp>
#include "../training_config.hpp"

namespace mlpack {

/**
 * Implementation of the Categorical Deep Q-Learning network.
 * For more information, see the following.
 *
 * @code
 * @misc{bellemare2017distributional,
 *   author  = {Marc G. Bellemare, Will Dabney, Rémi Munos},
 *   title   = {A Distributional Perspective on Reinforcement Learning},
 *   year    = {2017},
 *   url     = {http://arxiv.org/abs/1707.06887}
 * }
 * @endcode
 *
 * @tparam OutputLayerType The output layer type of the network.
 * @tparam InitType The initialization type used for the network.
 * @tparam NetworkType The type of network used for simple dqn.
 */
template<
  typename OutputLayerType = EmptyLoss,
  typename InitType = GaussianInitialization,
  typename NetworkType = FFN<OutputLayerType, InitType>
>
class CategoricalDQN
{
 public:
  /**
   * Default constructor.
   */
  CategoricalDQN() :
      network(), atomSize(0), vMin(0.0), vMax(0.0), isNoisy(false)
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of CategoricalDQN class.
   *
   * @param h1 Number of neurons in hiddenlayer-1.
   * @param h2 Number of neurons in hiddenlayer-2.
   * @param outputDim Number of neurons in output layer.
   * @param config Hyper-parameters for categorical dqn.
   * @param isNoisy Specifies whether the network needs to be of type noisy.
   * @param init Specifies the initialization rule for the network.
   * @param outputLayer Specifies the output layer type for network.
   */
  CategoricalDQN(const int h1,
                 const int h2,
                 const int outputDim,
                 TrainingConfig config,
                 const bool isNoisy = false,
                 InitType init = InitType(),
                 OutputLayerType outputLayer = OutputLayerType()):
      network(outputLayer, init),
      atomSize(config.AtomSize()),
      vMin(config.VMin()),
      vMax(config.VMax()),
      isNoisy(isNoisy)
  {
    network.Add(new Linear(h1));
    network.Add(new ReLU());
    if (isNoisy)
    {
      noisyLayerIndex.push_back(network.Network().size());
      network.Add(new NoisyLinear(h2));
      network.Add(new ReLU());
      noisyLayerIndex.push_back(network.Network().size());
      network.Add(new NoisyLinear(outputDim * atomSize));
    }
    else
    {
      network.Add(new Linear(h2));
      network.Add(new ReLU());
      network.Add(new Linear(outputDim * atomSize));
    }
  }

  /**
   * Construct an instance of CategoricalDQN class from a pre-constructed
   * network.
   *
   * @param network The network to be used by CategoricalDQN class.
   * @param config Hyper-parameters for categorical dqn.
   * @param isNoisy Specifies whether the network needs to be of type noisy.
   */
  CategoricalDQN(NetworkType& network,
                 TrainingConfig config,
                 const bool isNoisy = false):
      network(std::move(network)),
      atomSize(config.AtomSize()),
      vMin(config.VMin()),
      vMax(config.VMax()),
      isNoisy(isNoisy)
  { /* Nothing to do here. */ }

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * output layer function.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param state Input state.
   * @param actionValue Matrix to put output action values of states input.
   */
  void Predict(const arma::mat state, arma::mat& actionValue)
  {
    arma::mat q_atoms;
    network.Predict(state, q_atoms);
    activations.copy_size(q_atoms);
    actionValue.set_size(q_atoms.n_rows / atomSize, q_atoms.n_cols);
    arma::rowvec support = arma::linspace<arma::rowvec>(vMin, vMax, atomSize);
    for (size_t i = 0; i < q_atoms.n_rows; i += atomSize)
    {
      arma::mat activation = activations.rows(i, i + atomSize - 1);
      arma::mat input = q_atoms.rows(i, i + atomSize - 1);
      softMax.Forward(input, activation);
      activations.rows(i, i + atomSize - 1) = activation;
      actionValue.row(i/atomSize) = support * activation;
    }
  }

  /**
   * Perform the forward pass of the states in real batch mode.
   *
   * @param state The input state.
   * @param dist The predicted distributions.
   */
  void Forward(const arma::mat state, arma::mat& dist)
  {
    arma::mat q_atoms;
    network.Forward(state, q_atoms);
    activations.copy_size(q_atoms);
    for (size_t i = 0; i < q_atoms.n_rows; i += atomSize)
    {
      arma::mat activation = activations.rows(i, i + atomSize - 1);
      arma::mat input = q_atoms.rows(i, i + atomSize - 1);
      softMax.Forward(input, activation);
      activations.rows(i, i + atomSize - 1) = activation;
    }
    dist = activations;
  }

  /**
   * Resets the parameters of the network.
   */
  void Reset(const size_t inputDimensionality = 0)
  {
    network.Reset(inputDimensionality);
  }

  /**
   * Resets noise of the network, if the network is of type noisy.
   */
  void ResetNoise()
  {
    for (size_t i = 0; i < noisyLayerIndex.size(); ++i)
    {
      dynamic_cast<NoisyLinear*>(
          (network.Network()[noisyLayerIndex[i]]))->ResetNoise();
    }
  }

  //! Return the Parameters.
  const arma::mat& Parameters() const { return network.Parameters(); }
  //! Modify the Parameters.
  arma::mat& Parameters() { return network.Parameters(); }

  /**
   * Perform the backward pass of the state in real batch mode.
   *
   * @param state The input state.
   * @param lossGradients The loss gradients.
   * @param gradient The gradient.
   */
  void Backward(const arma::mat state,
                arma::mat& lossGradients,
                arma::mat& gradient)
  {
    arma::mat activationGradients(arma::size(activations));
    for (size_t i = 0; i < activations.n_rows; i += atomSize)
    {
      arma::mat activationGrad;
      arma::mat lossGrad = lossGradients.rows(i, i + atomSize - 1);
      arma::mat activation = activations.rows(i, i + atomSize - 1);
      softMax.Backward({} /* unused */, activation, lossGrad, activationGrad);
      activationGradients.rows(i, i + atomSize - 1) = activationGrad;
    }
    network.Backward(state, activationGradients, gradient);
  }

 private:
  //! Locally-stored network.
  NetworkType network;

  //! Locally-stored number of atoms.
  size_t atomSize;

  //! Locally-stored minimum value of support.
  double vMin;

  //! Locally-stored maximum value of support.
  double vMax;

  //! Locally-stored check for noisy network.
  bool isNoisy;

  //! Locally-stored indexes of noisy layers in the network.
  std::vector<size_t> noisyLayerIndex;

  //! Locally-stored softmax activation function.
  Softmax softMax;

  //! Locally-stored activations from softMax.
  arma::mat activations;
};

} // namespace mlpack

#endif
