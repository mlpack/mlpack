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

namespace mlpack {
namespace rl {

using namespace mlpack::ann;

/**
 * @tparam NetworkType The type of network used for dueling dqn.
 */
template <typename NetworkType = FFN<EmptyLoss<>,
                                    GaussianInitialization>>
class DuelingDQN
{
 public:
  /**
   * Default constructor.
   */
  DuelingDQN() : featureNetwork(), advantageNetwork(), valueNetwork()
  { /* Nothing to do here. */ }

  DuelingDQN(NetworkType featureNetwork,
             NetworkType advantageNetwork,
             NetworkType valueNetwork):
      featureNetwork(std::move(featureNetwork)),
      advantageNetwork(std::move(advantageNetwork)),
      valueNetwork(std::move(valueNetwork))
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
    arma::mat features, advantage, value;
    featureNetwork.Predict(state, features);
    advantageNetwork.Predict(features, advantage);
    valueNetwork.Predict(features, value);
    actionValue = advantage.each_col() + (value - arma::mean(advantage, 1));
  }

  /**
   * Perform the forward pass of the states in real batch mode.
   *
   * @param state The input state.
   * @param target The predicted target.
   */
  void Forward(const arma::mat state, arma::mat& target)
  {
    arma::mat features, advantage, value;
    featureNetwork.Forward(state, features);
    advantageNetwork.Forward(features, advantage);
    valueNetwork.Forward(features, value);
    actionValue = advantage.each_col() + (value - arma::mean(advantage, 1));
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
    featureNetwork.Backward(state, target, gradient);
  }

  /**
   * Resets the parameters of the network.
   */
  void ResetParameters()
  {
    featureNetwork.ResetParameters();
    advantageNetwork.ResetParameters();
    valueNetwork.ResetParameters();
  }

  //! Return the Parameters.
  const arma::mat& Parameters() const { return featureNetwork.Parameters(); }
  //! Modify the Parameters.
  arma::mat& Parameters() { return featureNetwork.Parameters(); }

 private:
  //! Locally-stored feature network.
  NetworkType featureNetwork;

  //! Locally-stored advantage network.
  NetworkType advantageNetwork;

  //! Locally-stored value network.
  NetworkType valueNetwork;
};

} // namespace rl
} // namespace mlpack

#endif
