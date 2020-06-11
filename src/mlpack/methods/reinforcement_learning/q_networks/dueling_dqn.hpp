/**
 * @file methods/reinforcement_learning/q_networks/dueling_dqn.hpp
 * @author Nishant Kumar
 *
 * This file contains the implementation of the dueling deep q network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_DUELING_DQN_HPP
#define MLPACK_METHODS_RL_DUELING_DQN_HPP

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
 * Implementation of the Dueling Deep Q-Learning network.
 * For more information, see the following.
 *
 * @code
 * @misc{wang2015dueling,
 *   author  = {Ziyu Wang, Tom Schaul, Matteo Hessel,Hado van Hasselt,
 *              Marc Lanctot, Nando de Freitas},
 *   title   = {Dueling Network Architectures for Deep Reinforcement Learning},
 *   year    = {2015},
 *   url     = {https://arxiv.org/abs/1511.06581}
 * }
 * @endcode
 * 
 * @tparam CompleteNetworkType The type of network used for full dueling dqn.
 * @tparam FeatureNetworkType The type of network used for feature network.
 * @tparam AdvantageNetworkType The type of network used for advantage network.
 * @tparam ValueNetworkType The type of network used for value network.
 */
template <
  typename CompleteNetworkType = FFN<EmptyLoss<>, GaussianInitialization>,
  typename FeatureNetworkType = Sequential<>,
  typename AdvantageNetworkType = Sequential<>,
  typename ValueNetworkType = Sequential<>
>
class DuelingDQN
{
 public:
  //! Default constructor.
  DuelingDQN() : isNoisy(false)
  {
    featureNetwork = new Sequential<>();
    valueNetwork = new Sequential<>();
    advantageNetwork = new Sequential<>();
    concat = new Concat<>(true);

    concat->Add(valueNetwork);
    concat->Add(advantageNetwork);
    completeNetwork.Add(new IdentityLayer<>());
    completeNetwork.Add(featureNetwork);
    completeNetwork.Add(concat);
  }

  /**
   * Construct an instance of DuelingDQN class.
   *
   * @param inputDim Number of inputs.
   * @param h1 Number of neurons in hiddenlayer-1.
   * @param h2 Number of neurons in hiddenlayer-2.
   * @param outputDim Number of neurons in output layer.
   * @param isNoisy Specifies whether the network needs to be of type noisy.
   */
  DuelingDQN(const int inputDim,
             const int h1,
             const int h2,
             const int outputDim,
             const bool isNoisy = false):
      completeNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.001)),
      isNoisy(isNoisy)
  {
    featureNetwork = new Sequential<>();
    featureNetwork->Add(new Linear<>(inputDim, h1));
    featureNetwork->Add(new ReLULayer<>());

    valueNetwork = new Sequential<>();
    advantageNetwork = new Sequential<>();

    if (isNoisy)
    {
      noisyLayerIndex.push_back(valueNetwork->Model().size());
      valueNetwork->Add(new NoisyLinear<>(h1, h2));
      advantageNetwork->Add(new NoisyLinear<>(h1, h2));

      valueNetwork->Add(new ReLULayer<>());
      advantageNetwork->Add(new ReLULayer<>());

      noisyLayerIndex.push_back(valueNetwork->Model().size());
      valueNetwork->Add(new NoisyLinear<>(h2, 1));
      advantageNetwork->Add(new NoisyLinear<>(h2, outputDim));
    }
    else
    {
      valueNetwork->Add(new Linear<>(h1, h2));
      valueNetwork->Add(new ReLULayer<>());
      valueNetwork->Add(new Linear<>(h2, 1));

      advantageNetwork->Add(new Linear<>(h1, h2));
      advantageNetwork->Add(new ReLULayer<>());
      advantageNetwork->Add(new Linear<>(h2, outputDim));
    }

    concat = new Concat<>(true);
    concat->Add(valueNetwork);
    concat->Add(advantageNetwork);

    completeNetwork.Add(new IdentityLayer<>());
    completeNetwork.Add(featureNetwork);
    completeNetwork.Add(concat);
    this->ResetParameters();
  }

  DuelingDQN(FeatureNetworkType featureNetwork,
             AdvantageNetworkType advantageNetwork,
             ValueNetworkType valueNetwork,
             const bool isNoisy = false):
      featureNetwork(std::move(featureNetwork)),
      advantageNetwork(std::move(advantageNetwork)),
      valueNetwork(std::move(valueNetwork)),
      isNoisy(isNoisy)
  {
    concat = new Concat<>(true);
    concat->Add(valueNetwork);
    concat->Add(advantageNetwork);
    completeNetwork.Add(new IdentityLayer<>());
    completeNetwork.Add(featureNetwork);
    completeNetwork.Add(concat);
    this->ResetParameters();
  }

  //! Copy constructor.
  DuelingDQN(const DuelingDQN& model) : isNoisy(false)
  { /* Nothing to do here. */ }

  //! Copy assignment operator.
  void operator = (const DuelingDQN& model)
  {
    *valueNetwork = *model.valueNetwork;
    *advantageNetwork = *model.advantageNetwork;
    *featureNetwork = *model.featureNetwork;
    isNoisy = model.isNoisy;
    noisyLayerIndex = model.noisyLayerIndex;
  }

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
    arma::mat advantage, value, networkOutput;
    completeNetwork.Predict(state, networkOutput);
    value = networkOutput.row(0);
    advantage = networkOutput.rows(1, networkOutput.n_rows - 1);
    actionValue = advantage.each_row() +
        (value - arma::mean(advantage));
  }

  /**
   * Perform the forward pass of the states in real batch mode.
   *
   * @param state The input state.
   * @param actionValue Matrix to put output action values of states input.
   */
  void Forward(const arma::mat state, arma::mat& actionValue)
  {
    arma::mat advantage, value, networkOutput;
    completeNetwork.Forward(state, networkOutput);
    value = networkOutput.row(0);
    advantage = networkOutput.rows(1, networkOutput.n_rows - 1);
    actionValue = advantage.each_row() +
        (value - arma::mean(advantage));
    this->actionValues = actionValue;
  }

  /**
   * Perform the backward pass of the state in real batch mode.
   *
   * @param state The input state.
   * @param target The training target.
   * @param gradient The gradient.
   */
  void Backward(const arma::mat state, arma::mat& target, arma::mat& gradient)
  {
    arma::mat gradLoss;
    lossFunction.Backward(this->actionValues, target, gradLoss);

    arma::mat gradValue = arma::sum(gradLoss);
    arma::mat gradAdvantage = gradLoss.each_row() - arma::mean(gradLoss);

    arma::mat grad = arma::join_cols(gradValue, gradAdvantage);
    completeNetwork.Backward(state, grad, gradient);
  }

  /**
   * Resets the parameters of the network.
   */
  void ResetParameters()
  {
    completeNetwork.ResetParameters();
  }

  /**
   * Resets noise of the network, if the network is of type noisy.
   */
  void ResetNoise()
  {
    for (size_t i = 0; i < noisyLayerIndex.size(); i++)
    {
      boost::get<NoisyLinear<>*>
          (valueNetwork->Model()[noisyLayerIndex[i]])->ResetNoise();
      boost::get<NoisyLinear<>*>
          (advantageNetwork->Model()[noisyLayerIndex[i]])->ResetNoise();
    }
  }

  //! Return the Parameters.
  const arma::mat& Parameters() const { return completeNetwork.Parameters(); }
  //! Modify the Parameters.
  arma::mat& Parameters() { return completeNetwork.Parameters(); }

 private:
  //! Locally-stored complete network.
  CompleteNetworkType completeNetwork;

  //! Locally-stored concat network.
  Concat<>* concat;

  //! Locally-stored feature network.
  FeatureNetworkType* featureNetwork;

  //! Locally-stored advantage network.
  AdvantageNetworkType* advantageNetwork;

  //! Locally-stored value network.
  ValueNetworkType* valueNetwork;

  //! Locally-stored check for noisy network.
  bool isNoisy;

  //! Locally-stored indexes of noisy layers in the network.
  std::vector<size_t> noisyLayerIndex;

  //! Locally-stored actionValues of the network.
  arma::mat actionValues;

  //! Locally-stored loss function.
  MeanSquaredError<> lossFunction;
};

} // namespace rl
} // namespace mlpack

#endif
