/**
 * @file episode_replay.hpp
 * @author Rohan Raj
 *
 * This file is an implementation of episode replay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_EPISODE_REPLAY_HPP
#define MLPACK_METHODS_RL_EPISODE_REPLAY_HPP

#include <math.h>
#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation of Episode Memory. 
 *
 * At each time step, interactions between the agent and the
 * environment will be saved to a memory buffer. When done
 * is true, we can simply sample experiences from the buffer 
 * to learn and train the agent. 
 *
 * For more information, see the following.
 *
 * @tparam EnvironmentType Desired task.
 */
template <typename EnvironmentType>
class EpisodeMemory
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  /**
   * Construct an instance of random experience replay class.
   *
   * @param capacity Total memory size in terms of number of steps in one episode.
   * @param dimension The dimension of an encoded state.
   */
  EpisodeMemory(const size_t capacity,
               const size_t dimension = StateType::dimension) :
      capacity(capacity),
      position(0),
      states(dimension, capacity),
      actions(capacity),
      rewards(capacity),
      nextStates(dimension, capacity),
      isTerminal(capacity),
      full(false)
  { /* Nothing to do here. */ }

  /**
   * Store the given experience.
   *
   * @param state Given state.
   * @param action Given action.
   * @param reward Given reward.
   * @param nextState Given next state.
   * @param isEnd Whether this state is terminal state.
   */
  void StoreEpisode(const StateType& state,
             ActionType action,
             double reward,
             const StateType& nextState,
             bool isEnd,
             double lambda = 0.99)

  {
    states.col(position) = state.Encode();
    actions(position) = action;
    // Adding discounted rewards.
    for ( size_t count = 0; count<position; count++ )
    {
      rewards(count) = rewards(count) + reward*pow(lambda, position-count);
    }
    rewards(position) = reward;
    nextStates.col(position) = nextState.Encode();
    isTerminal(position) = isEnd;
    position++;
    if (position == capacity || isEnd)
    {
      // capacity will try to end episode after a number of steps
      isTerminal(position-1) = true;
      full = true;
      position = 0;
    }
  }

  /**
   * @param sampledStates Sampled encoded states.
   * @param sampledActions Sampled actions.
   * @param sampledReturn Sampled return.
   * @param isTerminal Indicate whether corresponding next state is terminal
   *        state.
   *
   */
  void EpisodeReplay(arma::mat& sampledStates,
              arma::icolvec& sampledActions,
              arma::colvec& sampledReturn,
              arma::icolvec& isTerminal)
  {
    sampledStates = states;
    sampledActions = actions;
    sampledReturn = rewards;
    isTerminal = this->isTerminal;
  }
  /**
   * Get the number of transitions in the memory.
   *
   * @return Actual used memory size
   */
  const size_t& Size()
  {
    return full ? capacity : position;
  }

 private:
  //! Locally-stored total memory limit.
  size_t capacity;

  //! Indicate the position to store new transition.
  //! This also checks the number of steps taken in episode.
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

  //! Locally-stored indicator that whether the memory is full or not
  bool full;
};

} // namespace rl
} // namespace mlpack

#endif
