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

namespace mlpack {
namespace rl {

using namespace mlpack::ann;

/**
 * @tparam NetworkType The type of network used for categorical dqn.
 */
template <typename NetworkType = FFN<EmptyLoss<>, GaussianInitialization>>
class CategoricalDQN
{
 public:
  /**
   * Default constructor.
   */
  CategoricalDQN() : network(), isNoisy(false), atomSize(0) 
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of CategoricalDQN class.
   *
   * @param inputDim Number of inputs.
   * @param h1 Number of neurons in hiddenlayer-1.
   * @param h2 Number of neurons in hiddenlayer-2.
   * @param outputDim Number of neurons in output layer.
   */
  CategoricalDQN(const int inputDim,
            const int h1,
            const int h2,
            const int outputDim,
            const bool isNoisy = false,
            const size_t atomSize = 51):
      network(EmptyLoss<>(), GaussianInitialization(0, 0.001)),
      isNoisy(isNoisy),
      atomSize(atomSize)
  {
    network.Add(new Linear<>(inputDim, h1));
    network.Add(new ReLULayer<>());
    if(isNoisy)
    { 
      noisyLayerIndex.push_back(network.Model().size());
      network.Add(new NoisyLinear<>(h1, h2));
      network.Add(new ReLULayer<>());
      noisyLayerIndex.push_back(network.Model().size());
      network.Add(new NoisyLinear<>(h2, outputDim * atomSize));
    }
    else
    {
      network.Add(new Linear<>(h1, outputDim * atomSize));
    }
  }

  CategoricalDQN(NetworkType network, const bool isNoisy, size_t atomSize):
      network(std::move(network)),
      isNoisy(isNoisy),
      atomSize(atomSize)
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
    std::cout << "network params:" <<  network.Parameters() << std::endl;
    network.Predict(state, q_atoms);
    std::cout << "q_atoms:" <<  q_atoms << std::endl;
    activations.copy_size(q_atoms);
    actionValue.set_size(q_atoms.n_rows / atomSize, q_atoms.n_cols);
    double vMin = 0, vMax = 200.0;
    arma::rowvec support = arma::linspace<arma::rowvec>(vMin, vMax, atomSize);
    for(size_t i = 0; i < q_atoms.n_rows; i += atomSize)
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
    for(size_t i = 0; i < q_atoms.n_rows; i += atomSize)
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
  void ResetParameters()
  {
    network.ResetParameters();
  }

  /**
   * Resets noise of the network, is the network is of type noisy.
   */
  void ResetNoise()
  {
    for(size_t i = 0; i < noisyLayerIndex.size(); i++)
    {
      boost::get<NoisyLinear<>*>
          (network.Model()[noisyLayerIndex[i]])->ResetNoise();
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
   * @param lossGardients The loss gradients.
   * @param gradient The gradient.
   */
  void Backward(const arma::mat state, arma::mat& lossGradients, arma::mat& gradient)
  {
    arma::mat activationGradients(arma::size(activations));
    for(size_t i = 0; i < activations.n_rows; i += atomSize)
    {
      arma::mat activationGrad;
      arma::mat lossGrad = lossGradients.rows(i, i + atomSize - 1);
      arma::mat activation = activations.rows(i, i + atomSize - 1);
      softMax.Backward(activation, lossGrad, activationGrad);
      activationGradients.rows(i, i + atomSize - 1) = activationGrad;
    }
    network.Backward(state, activationGradients, gradient);
  }

 private:
  //! Locally-stored network.
  NetworkType network;

  //! Locally-stored check for noisy network.
  bool isNoisy;

  //! Locally-stored number of atoms.
  size_t atomSize;

  //! Locally-stored indexes of noisy layers in the network.
  std::vector<size_t> noisyLayerIndex;

  //! Locally-stored softmax activation function.
  Softmax<> softMax;

  //! Locally-stored activations from softMax.
  arma::mat activations;
};

} // namespace rl
} // namespace mlpack

#endif
