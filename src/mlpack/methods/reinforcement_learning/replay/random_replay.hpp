/**
 * @file random_replay.hpp
 * @author Shangtong Zhang and Rohan Raj
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

#include <math.h>
#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation of random experience replay.
 *
 * At each time step, interactions between the agent and the
 * environment will be saved to a memory buffer. When necessary,
 * we can simply sample previous experiences from the buffer to
 * train the agent. Typically this would be a random sample and
 * the memory will be a First-In-First-Out buffer.
 *
 * For more information, see the following.
 *
 * @code
 * @phdthesis{lin1993reinforcement,
 *  title  = {Reinforcement learning for robots using neural networks},
 *  author = {Lin, Long-Ji},
 *  year   = {1993},
 *  school = {Fujitsu Laboratories Ltd}
 * }
 * @endcode
 *
 * @tparam EnvironmentType Desired task.
 */
template <typename EnvironmentType>
class RandomReplay
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  /**
   * Construct an instance of random experience replay class.
   *
   * @param batchSize Number of examples returned at each sample.
   * @param capacity Total memory size in terms of number of examples.
   * @param dimension The dimension of an encoded state.
   */
  RandomReplay(const size_t batchSize,
               const size_t capacity,
               const size_t dimension = StateType::dimension) :
      batchSize(batchSize),
      capacity(capacity),
      position(0),
      states(dimension, capacity),
      actions(capacity),
      rewards(capacity),
      nextStates(dimension, capacity),
      isTerminal(capacity),
      full(false),
      steps(0)
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
  void Store(const StateType& state,
             ActionType action,
             double reward,
             const StateType& nextState,
             bool isEnd)

  {
    states.col(position) = state.Encode();
    actions(position) = action;
    rewards(position) = reward;
    nextStates.col(position) = nextState.Encode();
    isTerminal(position) = isEnd;
    position++;
    if (position == capacity)
    {
      full = true;
      position = 0;
    }
  }

  void StoreEpisode(const StateType& state,
             ActionType action,
             double reward,
             const StateType& nextState,
             bool isEnd,
             double lambda=0.99)

  {
    states.col(position) = state.Encode();
    actions(position) = action;
    rewards = rewards + reward*pow(lambda,steps);
    rewards(position) = reward;
    nextStates.col(position) = nextState.Encode();
    isTerminal(position) = isEnd;
    position++;
    steps++;
    if (position == capacity || isEnd)
    { 
      isTerminal(position-1)=true; // capacity will try to end episode after a number of steps
      steps = 0;
      full = true;
      position = 0;
    }
  }
  /**
   * Sample some experiences.
   *
   * @param sampledStates Sampled encoded states.
   * @param sampledActions Sampled actions.
   * @param sampledRewards Sampled rewards.
   * @param sampledNextStates Sampled encoded next states.
   * @param isTerminal Indicate whether corresponding next state is terminal
   *        state.
   */
  void Sample(arma::mat& sampledStates,
              arma::icolvec& sampledActions,
              arma::colvec& sampledRewards,
              arma::mat& sampledNextStates,
              arma::icolvec& isTerminal)
  {
    size_t upperBound = full ? capacity : position;
    arma::uvec sampledIndices = arma::randi<arma::uvec>(
        batchSize, arma::distr_param(0, upperBound - 1));

    sampledStates = states.cols(sampledIndices);
    sampledActions = actions.elem(sampledIndices);
    sampledRewards = rewards.elem(sampledIndices);
    sampledNextStates = nextStates.cols(sampledIndices);
    isTerminal = this->isTerminal.elem(sampledIndices);
  }
  /**
   * Sample some experiences for policy gradient.
   * Unlike usual methods this shares advantage for gradient ascent.
   * Reference : https://karpathy.github.io/2016/05/31/rl/
   *
   * @param sampledStates Sampled encoded states.
   * @param sampledActions Sampled actions.
   * @param sampledAdvantage Sampled advantage.
   * @param isTerminal Indicate whether corresponding next state is terminal
   *        state.
   */
  void EpisodeReplay(arma::mat& sampledStates,
              arma::icolvec& sampledActions,
              arma::colvec& sampledAdvantage,
              arma::icolvec& isTerminal)
  {
    // size_t upperBound = full ? capacity : position;
    // arma::uvec sampledIndices = arma::randi<arma::uvec>(
    //     batchSize, arma::distr_param(0, upperBound - 1));

    // sampledStates = states.cols(sampledIndices);
    // sampledActions = actions.elem(sampledIndices);
    // sampledAdvantage = rewards.elem(sampledIndices);
    // isTerminal = this->isTerminal.elem(sampledIndices);
    sampledStates = states;
    sampledActions = actions;
    sampledAdvantage = rewards;
    isTerminal = this->isTerminal;
    /*Advantage checks whether a particular action is good or not
    * Hence we need to check if it is better than the moving average of
    * the provious action. To do so we need to check how good it is from the 
    * average of the rewards recieved.
    */
    // mean= (mean*totalstepsvisited + sampledAdvantage(0))/(++totalstepsvisited);
    // sampledAdvantage = (sampledAdvantage - mean);
    // sampledAdvantage = sampledAdvantage / arma::stddev(rewards);
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
  
  
  //! Locally-stored number of examples of each sample.
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

  //! Locally-stored indicator that whether the memory is full or not
  bool full;

  // size_t totalstepsvisited;

  size_t steps;

};

} // namespace rl
} // namespace mlpack

#endif
