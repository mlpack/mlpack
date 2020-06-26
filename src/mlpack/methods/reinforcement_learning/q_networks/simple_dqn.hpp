/**
 * @file methods/reinforcement_learning/q_networks/simple_dqn.hpp
 * @author Nishant Kumar
 *
 * This file contains the implementation of the simple deep q network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_SIMPLE_DQN_HPP
#define MLPACK_METHODS_RL_SIMPLE_DQN_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

namespace mlpack {
namespace rl {

using namespace mlpack::ann;

/**
 * @tparam NetworkType The type of network used for simple dqn.
 */
template <typename NetworkType = FFN<MeanSquaredError<>,
                                    GaussianInitialization>>
class SimpleDQN
{
 public:
  /**
   * Default constructor.
   */
  SimpleDQN() : network(), isNoisy(false)
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of SimpleDQN class.
   *
   * @param inputDim Number of inputs.
   * @param h1 Number of neurons in hiddenlayer-1.
   * @param h2 Number of neurons in hiddenlayer-2.
   * @param outputDim Number of neurons in output layer.
   * @param isNoisy Specifies whether the network needs to be of type noisy.
   */
  SimpleDQN(const int inputDim,
            const int h1,
            const int h2,
            const int outputDim,
            const bool isNoisy = false):
      network(MeanSquaredError<>(), GaussianInitialization(0, 0.001)),
      isNoisy(isNoisy)
  {
    network.Add(new Linear<>(inputDim, h1));
    network.Add(new ReLULayer<>());
    if (isNoisy)
    {
      noisyLayerIndex.push_back(network.Model().size());
      network.Add(new NoisyLinear<>(h1, h2));
      network.Add(new ReLULayer<>());
      noisyLayerIndex.push_back(network.Model().size());
      network.Add(new NoisyLinear<>(h2, outputDim));
    }
    else
    {
      network.Add(new Linear<>(h1, h2));
      network.Add(new ReLULayer<>());
      network.Add(new Linear<>(h2, outputDim));
    }
  }

  SimpleDQN(NetworkType network, const bool isNoisy = false):
      network(std::move(network)),
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
    network.Predict(state, actionValue);
  }

  /**
   * Perform the forward pass of the states in real batch mode.
   *
   * @param state The input state.
   * @param target The predicted target.
   */
  void Forward(const arma::mat state, arma::mat& target)
  {
    network.Forward(state, target);
  }

  /**
   * Resets the parameters of the network.
   */
  void ResetParameters()
  {
    network.ResetParameters();
  }

  /**
   * Resets noise of the network, if the network is of type noisy.
   */
  void ResetNoise()
  {
    for (size_t i = 0; i < noisyLayerIndex.size(); i++)
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
   * @param target The training target.
   * @param gradient The gradient.
   */
  void Backward(const arma::mat state, arma::mat& target, arma::mat& gradient)
  {
    network.Backward(state, target, gradient);
  }

 private:
  //! Locally-stored network.
  NetworkType network;

  //! Locally-stored check for noisy network.
  bool isNoisy;

  //! Locally-stored indexes of noisy layers in the network.
  std::vector<size_t> noisyLayerIndex;
};

} // namespace rl
} // namespace mlpack

#endif
