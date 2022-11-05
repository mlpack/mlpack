/**
 * @file methods/reinforcement_learning/replay/hindsight_replay.hpp
 * @author Eshaan Agarwal
 *
 * This file is an implementation of hindsight experience replay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_HINDSIGHT_REPLAY_HPP
#define MLPACK_METHODS_RL_HINDSIGHT_REPLAY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of hindsight experience replay. Hindsight experience
 * replay is motivated by the human ability to learn useful things 
 * from failed attempts. It allows sample-efficient learning in 
 * sparse and binary reward situations.
 *
 * @code
 * @article{marcin2017hindsight,
 *  title   = {Hindsight Experience Replay},
 *  author  = {Marcin Andrychowicz, Filip Wolski, Alex Ray,
 *             Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew,
 *             Josh Tobin, Pieter Abbeel, Wojciech Zaremba},
 *  journal = {arXiv preprint arXiv:1707.01495},
 *  year    = {2017}
 *  }
 * @endcode
 *
 * @tparam EnvironmentType Desired task.
 */
template <typename EnvironmentType>
class HindsightReplay
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  struct Transition
  {
    StateType state;
    ActionType action;
    double reward;
    StateType nextState;
    bool isEnd;
    StateType goal;
  };

  enum goalStrategy
  {
    FINAL,
    FUTURE,
    EPISODE,
  };

  /**
   * Default constructor.
   */
  HindsightReplay():
      batchSize(0),
      capacity(0),
      positionIdx(0),
      full(false),
      noOfHERTransitions(1),
      goalSelectionStrategy(goalStrategy::FUTURE),
      nSteps(0)
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of hindsight experience replay class.
   *
   * @param batchSize Number of examples returned at each sample.
   * @param capacity Total memory size in terms of number of examples.
   * @param noOfHERTransitions Number of HER Transitions per regular transitions
   * (data coming from normal experience replay in replay buffer)
   * @param nSteps Number of steps to look in the future.
   * @param dimension The dimension of an encoded state.
   * @param strategy goal selection strategy for HER
   */
  HindsightReplay(const size_t batchSize,
                    const size_t capacity,
                    const size_t noOfHERTransitions = 4,
                    EnvironmentType environment = EnvironmentType(),
                    goalStrategy strategy = goalStrategy::FUTURE,
                    const size_t nSteps = 1,
                    const size_t dimension = StateType::dimension) :
      batchSize(batchSize),
      capacity((noOfHERTransitions+1)*capacity),
      positionIdx(0),
      full(false),
      noOfHERTransitions(noOfHERTransitions),
      goalSelectionStrategy(strategy),
      nSteps(nSteps),
      states(dimension, (noOfHERTransitions+1)*capacity),
      actions((noOfHERTransitions+1)*capacity),
      rewards((noOfHERTransitions+1)*capacity),
      nextStates(dimension, (noOfHERTransitions+1)*capacity),
      goals(dimension, (noOfHERTransitions+1)*capacity),
      isTerminal((noOfHERTransitions+1)*capacity),
      environment(environment)
  { /* Nothing to do here. */ }

  /**
   * Sample goal according to goal strategy
   * 
   * @param transitionIndex index of current transition
   */
  StateType SampleGoal(const int transitionIndex)
  { 
    StateType desiredGoal;
    switch (goalSelectionStrategy)
    {
      case goalStrategy::FINAL :
      {
        // Sample goal as the final state of the epsiode
        desiredGoal = episodeTransitions.back().nextState;
        break;
      }
      case goalStrategy::FUTURE :
      {
        // Sample goal as random state from that transition to end of epsiode
        // size_t index = arma::randi(
        // arma::distr_param(transitionIndex+1, episodeTransitions.size() -1));
        desiredGoal = episodeTransitions[arma::randi(
        arma::distr_param(transitionIndex+1, episodeTransitions.size() -1))].nextState;
        break;
      }
      case goalStrategy::EPISODE :
      {
        // Sample goal as random state from the whole epsiode
        desiredGoal = episodeTransitions[arma::randi(
        arma::distr_param(0, episodeTransitions.size() -1))].nextState;
        break;
      }
    }
    return desiredGoal;
  }

  /**
   * Sample goals according to goal strategy and HER ratio
   * 
   * @param desiredGoals desired goals as per goal strategy
   * @param  transitionIndex index of current transition
   */
  std::vector<StateType> SampleGoals(int transitionIndex)
  { 
    std::vector<StateType> desiredGoals(noOfHERTransitions);
    // Store HER transitions based on strategy for each transition in epsiode
    for (size_t goalIndex = 0; goalIndex < noOfHERTransitions; ++goalIndex)
    {
      StateType desiredGoal = SampleGoal(transitionIndex);
      desiredGoals[goalIndex] = desiredGoal;
    }

    return desiredGoals;
  }

  /**
   * Sample goal according to goal strategy
   * @param discount The discount parameter.
   */
  void StoreHERTransitions(const double& discount)
  { 
    std::vector<Transition> baseTransitions = episodeTransitions;

    // cannot create a future hindsight goal in the last transition of episode
    if (goalSelectionStrategy == goalStrategy::FUTURE)
    {
      baseTransitions.pop_back();
    }

    for(size_t transitionIndex = 0; transitionIndex < baseTransitions.size();
        ++transitionIndex)
    { 
      // Sample goals for particular transition
      std::vector<StateType> desiredGoals = SampleGoals(transitionIndex);

      for(size_t goalIndex = 0; goalIndex < noOfHERTransitions; ++goalIndex)
      { 
        // Get reward for that particular HER transition
        double reward = environment.GetHERReward(baseTransitions[transitionIndex].nextState,
                                            desiredGoals[goalIndex]);

        // store transition in nStepBuffer
        nStepBuffer.push_back({baseTransitions[transitionIndex].state,
                              baseTransitions[transitionIndex].action,
                              reward,
                              baseTransitions[transitionIndex].nextState,
                              baseTransitions[transitionIndex].isEnd,
                              desiredGoals[goalIndex]});

        // Single step transition is not ready.
        if (nStepBuffer.size() < nSteps)
          return;

        // To keep the queue size fixed to nSteps.
        if (nStepBuffer.size() > nSteps)
          nStepBuffer.pop_front();

        // Before moving ahead, lets confirm if our fixed size buffer works.
        assert(nStepBuffer.size() == nSteps);

        // Make a n-step transition.
        GetNStepInfo(reward,
                    baseTransitions[transitionIndex].nextState,
                    baseTransitions[transitionIndex].isEnd,
                    discount);

        StateType state = nStepBuffer.front().state;
        ActionType action = nStepBuffer.front().action;
        states.col(positionIdx) = state.Encode();
        actions[positionIdx] = action;
        rewards(positionIdx) = reward;
        nextStates.col(positionIdx) = baseTransitions[transitionIndex].nextState.Encode();
        isTerminal(positionIdx) = false;
        positionIdx++;
        if (positionIdx == capacity)
        {
          full = true;
          positionIdx = 0;
        }
      }
    }
  }

  /**
   * Store the given experience and set the goal for the given experience.
   *
   * @param state Given state.
   * @param action Given action.
   * @param reward Given reward.
   * @param nextState Given next state.
   * @param isEnd Whether next state is terminal state.
   * @param goal goal of the given experience
   * @param discount The discount parameter.
   */
  void Store(StateType state,
             ActionType action,
             double reward,
             StateType nextState,
             bool isEnd,
             const double& discount,
             StateType goal)
  { 
    // Store all the transitions of episode in a vector
    episodeTransitions.push_back({state, action, reward, nextState, isEnd, goal});

    nStepBuffer.push_back({state, action, reward, nextState, isEnd, goal});

    // Single step transition is not ready.
    if (nStepBuffer.size() < nSteps)
      return;

    // To keep the queue size fixed to nSteps.
    if (nStepBuffer.size() > nSteps)
      nStepBuffer.pop_front();

    // Before moving ahead, lets confirm if our fixed size buffer works.
    assert(nStepBuffer.size() == nSteps);

    // Make a n-step transition.
    GetNStepInfo(reward, nextState, isEnd, discount);

    state = nStepBuffer.front().state;
    action = nStepBuffer.front().action;
    states.col(positionIdx) = state.Encode();
    actions[positionIdx] = action;
    rewards(positionIdx) = reward;
    nextStates.col(positionIdx) = nextState.Encode();
    isTerminal(positionIdx) = isEnd;
    positionIdx++;
    if (positionIdx == capacity)
    {
      full = true;
      positionIdx = 0;
    }
  }

  /**
   * Get the reward, next state and terminal boolean for nth step.
   *
   * @param reward Given reward.
   * @param nextState Given next state.
   * @param isEnd Whether next state is terminal state.
   * @param discount The discount parameter.
   */
  void GetNStepInfo(double& reward,
                    StateType& nextState,
                    bool& isEnd,
                    const double& discount)
  {
    reward = nStepBuffer.back().reward;
    nextState = nStepBuffer.back().nextState;
    isEnd = nStepBuffer.back().isEnd;

    // Should start from the second last transition in buffer.
    for (int i = nStepBuffer.size() - 2; i >= 0; i--)
    {
      bool iE = nStepBuffer[i].isEnd;
      reward = nStepBuffer[i].reward + discount * reward * (1 - iE);
      if (iE)
      {
        nextState = nStepBuffer[i].nextState;
        isEnd = iE;
      }
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
              std::vector<ActionType>& sampledActions,
              arma::rowvec& sampledRewards,
              arma::mat& sampledNextStates,
              arma::irowvec& isTerminal)
  {
    size_t upperBound = full ? capacity : positionIdx;
    arma::uvec sampledIndices = arma::randi<arma::uvec>(
        batchSize, arma::distr_param(0, upperBound - 1));

    sampledStates = states.cols(sampledIndices);
    for (size_t t = 0; t < sampledIndices.n_rows; t ++)
      sampledActions.push_back(actions[sampledIndices[t]]);
    sampledRewards = rewards.elem(sampledIndices).t();
    sampledNextStates = nextStates.cols(sampledIndices);
    isTerminal = this->isTerminal.elem(sampledIndices).t();
  }

  /**
   * Get the number of transitions in the memory.
   *
   * @return Actual used memory size.
   */
  const size_t& Size()
  {
    return full ? capacity : positionIdx;
  }

  /**
   * Update the priorities of transitions and Update the gradients.
   *
   * @param target The learned value.
   * @param sampledActions Agent's sampled action.
   * @param nextActionValues Agent's next action.
   * @param gradients The model's gradients.
   */
  void Update(arma::mat target,
              std::vector<ActionType> sampledActions,
              arma::mat nextActionValues,
              arma::mat& gradients)
  {
    /* Do nothing for hindsight replay. */
  }

  //! Get the number of steps for n-step agent.
  const size_t& NSteps() const { return nSteps; }

 private:
  //! Locally-stored number of examples of each sample.
  size_t batchSize;

  //! Locally-stored total memory limit.
  size_t capacity;

  //! Indicate the position to store new transition.
  size_t positionIdx;

  //! Locally-stored indicator that whether the memory is full or not.
  bool full;

  //! Locally-stored number of steps to look into the future.
  size_t nSteps;

  //! Locally-stored buffer containing n consecutive steps.
  std::deque<Transition> nStepBuffer;

  //! Locally-stored encoded previous states.
  arma::mat states;

  //! Locally-stored previous actions.
  std::vector<ActionType> actions;

  //! Locally-stored previous rewards.
  arma::rowvec rewards;

  //! Locally-stored encoded previous next states.
  arma::mat nextStates;

  //! Locally-stored termination information of previous experience.
  arma::irowvec isTerminal;

  //! Locally-stored encoded goal states.
  arma::mat goals;

  //! Locally-stored transitions of episode
  std::vector<Transition> episodeTransitions;

  //! Number of HER transitions per Normal transitions
  //! (data coming from normal experience replay in replay buffer)
  size_t noOfHERTransitions;

  //! goal selection startegy for HER
  goalStrategy goalSelectionStrategy;

  //! Locally-stored reinforcement learning task.
  EnvironmentType environment;
};

} // namespace mlpack

#endif
