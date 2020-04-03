/**
 * @file episodic_replay.hpp
 * @author Narayanan E R
 *
 * This file is an implementation of episodic experience replay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_REPLAY_EPISODIC_REPLAY_HPP
#define MLPACK_METHODS_RL_REPLAY_EPISODIC_REPLAY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation of episodic experience replay.
 *
 * Each episode of interactions between the agent and the
 * environment will be saved to a memory buffer. When necessary,
 * we can simply sample previous episodes or the most recent episode
 * from the buffer to train the agent.
 *
 * @tparam EnvironmentType Desired task.
 */

template <typename EnvironmentType>
class EpisodicReplay
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  /**
  * Construct an instance of EpisodicReplay class.
  */
  EpisodicReplay():
      capacity(0),
      position(0),
      maxEpisodeLen(0),
      full(false)
  { /* Nothing to do here. */ }

  /**
  * Construct an instance of EpisodicReplay class.
  *
  * @param capacity Maximum number of episodes.
  * @param maxEpisode_len The maximum episode length possible.
  * @param dimension The dimension of an encoded state.
  */
  EpisodicReplay(const size_t capacity,
                 const size_t maxEpisodeLen) :
      capacity(capacity),
      position(0),
      maxEpisodeLen(maxEpisodeLen),
      full(false)
  {
    states.resize(capacity);
    next_states.resize(capacity);
    rewards.resize(capacity);
    actions.resize(capacity);
    isTerminal.resize(capacity);
  }

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
    if (clear)
    {
      states[position].clear();
      actions[position].clear();
      rewards[position].clear();
      next_states[position].clear();
      isTerminal[position].clear();
      clear = false;
    }

    states[position].push_back(state.Encode());
    actions[position].push_back(action);
    rewards[position].push_back(reward);
    next_states[position].push_back(nextState.Encode());
    isTerminal[position].push_back(isEnd);

    if (isEnd || states[position].size() == maxEpisodeLen)
    {
      position++;
      clear = true;
    }
    if (position == capacity)
    {
      full = true;
      position = 0;
    }
  }

  /**
  * Get the number of episodes in the memory.
  *
  * @return Actual used memory size
  */
  const size_t Size()
  {
    if (states[position].size() == 0)
      return full ? capacity : position;
    return full ? capacity : (position+1);
  }

  /**
  * Get the most recently added episode.
  *
  * @param sampledStates Sampled encoded states.
  * @param sampledActions Sampled actions.
  * @param sampledRewards Sampled rewards.
  * @param sampledNextStates Sampled encoded next states.
  * @param isTerminal Indicate whether corresponding next state is terminal
  *        state.
  * @param random Whether episode is sampled random or most recent episode
  *        is sampled
  */
  void Sample(arma::mat& episodeStates,
              arma::icolvec& episodeActions,
              arma::colvec& episodeRewards,
              arma::mat& episodeNextStates,
              arma::icolvec& isTerminal,
              bool random = false)
  {
    int episodeNum = 0;
    if (random)
    {
      size_t upperBound = full ? capacity : position;
      int high = upperBound ? upperBound : 1;
      episodeNum = math::RandInt(0, high);
    }
    else
    {
      if (states[position].size() == 0 || clear == true)
      {
        if (position == 0 && full)
            episodeNum = capacity-1;
        else if(position != 0)
          episodeNum = position-1;
      }
      else
        episodeNum = position;
    }
    int i = 0;
    for (auto state : states[episodeNum])
    {
      if (i == 0)
      {
        episodeStates = state;
        i++;
      }
      else
        episodeStates = arma::join_rows(episodeStates, state);
    }
    episodeActions = arma::conv_to<arma::icolvec>::from(actions[episodeNum]);
    episodeRewards = arma::conv_to<arma::colvec>::from(rewards[episodeNum]);
    i = 0;
    for (auto state : next_states[episodeNum])
    {
      if (i == 0)
      {
        episodeNextStates = state;
        i++;
      }
      else
        episodeNextStates = arma::join_rows(episodeNextStates, state);
    }
    isTerminal = arma::conv_to<arma::icolvec>::from(
        this->isTerminal[episodeNum]);
  }

  void Update(arma::mat /* target */,
              arma::icolvec /* sampledActions */,
              arma::mat /* nextActionValues */,
              arma::mat& /* gradients */)
  {
    /* Do nothing for episodic replay. */
  }

 private:
  //! Locally-stored total episode limit.
  size_t capacity;

  //! Indicate the position to store new episode.
  size_t position;

  //! Locally-stored maximum episode length
  size_t maxEpisodeLen;

  //! Locally-stored indicator whether to clear current position before storing
  bool clear;

  //! Locally-stored encoded previous states.
  std::vector< std::vector< arma::colvec> > states;

  //! Locally-stored previous actions.
  std::vector< std::vector<int> > actions;

  //! Locally-stored previous rewards.
  std::vector< std::vector<double> > rewards;

  //! Locally-stored encoded previous next states.
  std::vector< std::vector< arma::colvec> > next_states;

  //! Locally-stored termination information of previous experience.
  std::vector< std::vector<int> > isTerminal;

  //! Locally-stored indicator that whether the memory is full or not
  bool full;
};

} // namespace rl
} // namespace mlpack

#endif
