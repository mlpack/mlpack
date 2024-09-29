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
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/empty_loss.hpp>

namespace mlpack {

/**
 * Implementation of the Dueling Deep Q-Learning network.
 * For more information, see the following.
 *
 * @code
 * @misc{wang2015dueling,
 *   author  = {Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt,
 *              Marc Lanctot, Nando de Freitas},
 *   title   = {Dueling Network Architectures for Deep Reinforcement Learning},
 *   year    = {2015},
 *   url     = {https://arxiv.org/abs/1511.06581}
 * }
 * @endcode
 *
 * @tparam OutputLayerType The output layer type of the network.
 * @tparam InitType The initialization type used for the network.
 * @tparam CompleteNetworkType The type of network used for full dueling dqn.
 * @tparam FeatureNetworkType The type of network used for feature network.
 * @tparam AdvantageNetworkType The type of network used for advantage network.
 * @tparam ValueNetworkType The type of network used for value network.
 */
template <
  typename OutputLayerType = EmptyLoss,
  typename InitType = GaussianInitialization,
  typename CompleteNetworkType = FFN<OutputLayerType, InitType>,
  typename FeatureNetworkType = MultiLayer<arma::mat>,
  typename AdvantageNetworkType = MultiLayer<arma::mat>,
  typename ValueNetworkType = MultiLayer<arma::mat>
>
class DuelingDQN
{
 public:
  //! Default constructor.
  DuelingDQN() : isNoisy(false)
  {
    // TODO: this really ought to use a DAG network, but that's not implemented
    // yet.
    featureNetwork = new MultiLayer<arma::mat>();
    valueNetwork = new MultiLayer<arma::mat>();
    advantageNetwork = new MultiLayer<arma::mat>();
    concat = new Concat();

    concat->Add(valueNetwork);
    concat->Add(advantageNetwork);
    completeNetwork.Add(featureNetwork);
    completeNetwork.Add(concat);
  }

  /**
   * Construct an instance of DuelingDQN class.
   *
   * @param h1 Number of neurons in hiddenlayer-1.
   * @param h2 Number of neurons in hiddenlayer-2.
   * @param outputDim Number of neurons in output layer.
   * @param isNoisy Specifies whether the network needs to be of type noisy.
   * @param init Specifies the initialization rule for the network.
   * @param outputLayer Specifies the output layer type for network.
   */
  DuelingDQN(const int h1,
             const int h2,
             const int outputDim,
             const bool isNoisy = false,
             InitType init = InitType(),
             OutputLayerType outputLayer = OutputLayerType()):
      completeNetwork(outputLayer, init),
      isNoisy(isNoisy)
  {
    featureNetwork = new MultiLayer<arma::mat>();
    featureNetwork->Add(new Linear(h1));
    featureNetwork->Add(new ReLU());

    valueNetwork = new MultiLayer<arma::mat>();
    advantageNetwork = new MultiLayer<arma::mat>();

    if (isNoisy)
    {
      noisyLayerIndex.push_back(valueNetwork->Network().size());
      valueNetwork->Add(new NoisyLinear(h2));
      advantageNetwork->Add(new NoisyLinear(h2));

      valueNetwork->Add(new ReLU());
      advantageNetwork->Add(new ReLU());

      noisyLayerIndex.push_back(valueNetwork->Network().size());
      valueNetwork->Add(new NoisyLinear(1));
      advantageNetwork->Add(new NoisyLinear(outputDim));
    }
    else
    {
      valueNetwork->Add(new Linear(h2));
      valueNetwork->Add(new ReLU());
      valueNetwork->Add(new Linear(1));

      advantageNetwork->Add(new Linear(h2));
      advantageNetwork->Add(new ReLU());
      advantageNetwork->Add(new Linear(outputDim));
    }

    concat = new Concat();
    concat->Add(valueNetwork);
    concat->Add(advantageNetwork);

    completeNetwork.Add(featureNetwork);
    completeNetwork.Add(concat);
  }

  /**
   * Construct an instance of DuelingDQN class from a pre-constructed network.
   *
   * @param featureNetwork The feature network to be used by DuelingDQN class.
   * @param advantageNetwork The advantage network to be used by DuelingDQN class.
   * @param valueNetwork The value network to be used by DuelingDQN class.
   * @param isNoisy Specifies whether the network needs to be of type noisy.
   */
  DuelingDQN(FeatureNetworkType& featureNetwork,
             AdvantageNetworkType& advantageNetwork,
             ValueNetworkType& valueNetwork,
             const bool isNoisy = false):
      featureNetwork(featureNetwork),
      advantageNetwork(advantageNetwork),
      valueNetwork(valueNetwork),
      isNoisy(isNoisy)
  {
    concat = new Concat();
    concat->Add(valueNetwork);
    concat->Add(advantageNetwork);
    completeNetwork.Add(featureNetwork);
    completeNetwork.Add(concat);
  }

  //! Copy constructor.
  DuelingDQN(const DuelingDQN& /* model */) : isNoisy(false)
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
    actionValue = advantage.each_row() + (value - arma::mean(advantage));
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

    arma::mat gradValue = sum(gradLoss);
    arma::mat gradAdvantage = gradLoss.each_row() - arma::mean(gradLoss);

    arma::mat grad = join_cols(gradValue, gradAdvantage);
    completeNetwork.Backward(state, grad, gradient);
  }

  /**
   * Resets the parameters of the network.
   */
  void Reset(const size_t inputDimensionality = 0)
  {
    completeNetwork.Reset(inputDimensionality);
  }

  /**
   * Resets noise of the network, if the network is of type noisy.
   */
  void ResetNoise()
  {
    for (size_t i = 0; i < noisyLayerIndex.size(); i++)
    {
      dynamic_cast<NoisyLinear*>(
          (valueNetwork->Network()[noisyLayerIndex[i]]))->ResetNoise();
      dynamic_cast<NoisyLinear*>(
          (advantageNetwork->Network()[noisyLayerIndex[i]]))->ResetNoise();
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
  Concat* concat;

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
  MeanSquaredError lossFunction;
};

} // namespace mlpack

#endif
