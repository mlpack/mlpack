/**
 * @file methods/reinforcement_learning/q_networks/simple_dqn.hpp
 * @author Nishant Kumar
 *
 * This file contains the definition of the simple deep q network.
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
 * Implementation of simple linear Deep-Q-Learning model.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename OutputLayerType = MeanSquaredError<>,
  typename InitializationRuleType = RandomInitialization
>
class SimpleDQN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = FFN<OutputLayerType, InitializationRuleType>;

  //! Default constructor.
  SimpleDQN();

  /**
   * Construct an instance of SimpleDQN class.
   *
   * @param inputDim Number of inputs.
   * @param h1 Number of neurons in hidden layer-1.
   * @param h2 Number of neurons in hidden layer-2.
   * @param outputDim Number of neurons in output layer.
   */
  SimpleDQN(const size_t inputDim,
            const size_t h1,
            const size_t h2,
            const size_t outputDim);

  /**
   * Constructor for using a user-defined network.
   *
   * @param network User-defined Feed Forward network to be used for Deep
   *                Q-Learning.
   */
  SimpleDQN(NetworkType& network);

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
  void Predict(const arma::mat& state, arma::mat& actionValue);

  /**
   * Perform the forward pass of the states in real batch mode.
   *
   * @param state The input state.
   * @param target The predicted target.
   */
  void Forward(const arma::mat& state, arma::mat& target);

  /**
   * Perform the backward pass of the state in real batch mode.
   *
   * @param state The input state.
   * @param target The training target.
   * @param gradient The gradient.
   */
<<<<<<< HEAD
  void Backward(const arma::mat& state, arma::mat& target,
      arma::mat& gradient)
=======
  void Backward(const arma::mat& state,
                const arma::mat& target,
                arma::mat& gradient);

  /**
   * Resets the parameters of the network.
   */
  void ResetParameters();

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  //! Return the Parameters.
  const arma::mat& Parameters() const { return network.Parameters(); }
  //! Modify the Parameters.
  arma::mat& Parameters() { return network.Parameters(); }
>>>>>>> 8e7600689... Seperate Files

 private:
  //! Locally-stored network.
  NetworkType network;
};

} // namespace rl
} // namespace mlpack

// Include implementation.
#include "simple_dqn_impl.hpp"

#endif
