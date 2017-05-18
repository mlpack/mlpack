/**
 * @file random_replay.hpp
 * @author Shangtong Zhang
 *
 * This file is an implementation of random experience replay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_REPLAY_RANDOM_REPLAY_HPP
#define MLPACK_METHODS_RL_REPLAY_RANDOM_REPLAY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation of random experience replay.
 *
 * @tparam EnvironmentType Desired task.
 */
template <typename EnvironmentType>
class RandomReplay
{
 public:
  using ActionType = typename EnvironmentType::Action;
  using StateType = typename EnvironmentType::State;

  /**
   * Construct an instance of random experience replay class.
   *
   * @param batchSize # of examples returned at each sample.
   * @param capacity Total memory size in terms of # of examples.
   * @param dimension The dimension of an encoded state.
   */
  RandomReplay(size_t batchSize,
               size_t capacity,
               size_t dimension = StateType::dimension) :
      batchSize(batchSize),
      capacity(capacity),
      position(0),
      states(dimension, 0),
      nextStates(dimension, 0)
  { /* Nothing to do here. */ }

  /**
   * Store the given experience.
   *
   * @param state Given state.
   * @param action Given action.
   * @param reward Given reward.
   * @param nextState Given next state.
   * @param isEnd Whether next state is terminal state.
   */
  void Store(const StateType& state, ActionType action,
             double reward, const StateType& nextState, bool isEnd)
  {
    if (isTerminal.n_elem < capacity)
    {
      states.insert_cols(position, 1);
      actions.insert_rows(position, 1);
      rewards.insert_rows(position, 1);
      nextStates.insert_cols(position, 1);
      isTerminal.insert_rows(position, 1);
    }
    states.col(position) = state.Encode();
    actions(position) = action;
    rewards(position) = reward;
    nextStates.col(position) = nextState.Encode();
    isTerminal(position) = isEnd;
    position++;
    position %= capacity;
  }

  /**
   * Sample some experiences.
   *
   * @param sampledStates Sampled encoded states.
   * @param sampledActions Sampled actions.
   * @param sampledRewards Sampled rewards.
   * @param sampledNextStates Sampled encoded next states.
   * @param isTerminal Indicate whether corresponding next state is terminal state.
   */
  void Sample(arma::mat& sampledStates,
              arma::icolvec& sampledActions,
              arma::colvec& sampledRewards,
              arma::mat& sampledNextStates,
              arma::icolvec& isTerminal)
  {
    size_t upperBound = this->isTerminal.n_elem == capacity ? capacity : position;
    arma::uvec sampledIndices =
        arma::randi<arma::uvec>(batchSize, arma::distr_param(0, upperBound - 1));
    sampledStates = states.cols(sampledIndices);
    sampledActions = actions.elem(sampledIndices);
    sampledRewards = rewards.elem(sampledIndices);
    sampledNextStates = nextStates.cols(sampledIndices);
    isTerminal = this->isTerminal.elem(sampledIndices);
  }

  /**
   * Get the # of transitions in the memory.
   *
   * @return Actual memory size
   */
  size_t Size()
  {
    return isTerminal.n_elem == capacity ? capacity : position;
  }

 private:
  //! Locally-stored # of examples of each sample.
  size_t batchSize;

  //! Locally-stored total memory limit.
  size_t capacity;

  //! Indicate the position to store new transition.
  size_t position;

  //! Locally-stored encoded previous states.
  arma::mat states;

  //! Locally-stored previous actions.
  arma::icolvec actions;

  //! Locally-stored previous rewards.
  arma::colvec rewards;

  //! Locally-stored encoded previous next states.
  arma::mat nextStates;

  //! Locally-stored termination information of previous experience.
  arma::icolvec isTerminal;
};

} // namespace rl
} // namespace mlpack

#endif
