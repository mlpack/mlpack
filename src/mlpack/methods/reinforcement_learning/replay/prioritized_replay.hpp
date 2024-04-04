/**
 * @file methods/reinforcement_learning/replay/prioritized_replay.hpp
 * @author Xiaohong
 *
 * This file is an implementation of prioritized experience replay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_PRIORITIZED_REPLAY_HPP
#define MLPACK_METHODS_RL_PRIORITIZED_REPLAY_HPP

#include <mlpack/prereqs.hpp>
#include "sumtree.hpp"

namespace mlpack {

/**
 * Implementation of prioritized experience replay. Prioritized experience
 * replay can replay important transitions more frequently by prioritizing
 * transitions, and make agent learn more efficiently.
 *
 * @code
 * @article{schaul2015prioritized,
 *  title   = {Prioritized experience replay},
 *  author  = {Schaul, Tom and Quan, John and Antonoglou,
 *             Ioannis and Silver, David},
 *  journal = {arXiv preprint arXiv:1511.05952},
 *  year    = {2015}
 *  }
 * @endcode
 *
 * @tparam EnvironmentType Desired task.
 */
template <typename EnvironmentType>
class PrioritizedReplay
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
  };

  /**
   * Default constructor.
   */
  PrioritizedReplay():
      batchSize(0),
      capacity(0),
      position(0),
      full(false),
      alpha(0),
      maxPriority(0),
      initialBeta(0),
      beta(0),
      replayBetaIters(0),
      nSteps(0)
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of prioritized experience replay class.
   *
   * @param batchSize Number of examples returned at each sample.
   * @param capacity Total memory size in terms of number of examples.
   * @param alpha How much prioritization is used.
   * @param nSteps Number of steps to look in the future.
   * @param dimension The dimension of an encoded state.
   */
  PrioritizedReplay(const size_t batchSize,
                    const size_t capacity,
                    const double alpha,
                    const size_t nSteps = 1,
                    const size_t dimension = StateType::dimension) :
      batchSize(batchSize),
      capacity(capacity),
      position(0),
      full(false),
      alpha(alpha),
      maxPriority(1.0),
      initialBeta(0.6),
      replayBetaIters(10000),
      nSteps(nSteps),
      states(dimension, capacity),
      actions(capacity),
      rewards(capacity),
      nextStates(dimension, capacity),
      isTerminal(capacity)
  {
    size_t size = 1;
    while (size < capacity)
    {
      size *= 2;
    }

    beta = initialBeta;
    idxSum = SumTree<double>(size);
  }

  /**
   * Store the given experience and set the priorities for the given experience.
   *
   * @param state Given state.
   * @param action Given action.
   * @param reward Given reward.
   * @param nextState Given next state.
   * @param isEnd Whether next state is terminal state.
   * @param discount The discount parameter.
   */
  void Store(StateType state,
             ActionType action,
             double reward,
             StateType nextState,
             bool isEnd,
             const double& discount)
  {
    nStepBuffer.push_back({state, action, reward, nextState, isEnd});

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
    states.col(position) = state.Encode();
    actions[position] = action;
    rewards(position) = reward;
    nextStates.col(position) = nextState.Encode();
    isTerminal(position) = isEnd;

    idxSum.Set(position, maxPriority * alpha);

    position++;
    if (position == capacity)
    {
      full = true;
      position = 0;
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
   * Sample some experience according to their priorities.
   *
   * @return The indices to be chosen.
   */
  arma::ucolvec SampleProportional()
  {
    arma::ucolvec idxes(batchSize);
    double totalSum = idxSum.Sum(0, (full ? capacity : position));
    double sumPerRange = totalSum / batchSize;
    for (size_t bt = 0; bt < batchSize; bt++)
    {
      const double mass = arma::randu() * sumPerRange + bt * sumPerRange;
      idxes(bt) = idxSum.FindPrefixSum(mass);
    }
    return idxes;
  }

  /**
   * Sample some experience according to their priorities.
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
    sampledIndices = SampleProportional();
    BetaAnneal();

    sampledStates = states.cols(sampledIndices);
    for (size_t t = 0; t < sampledIndices.n_rows; t ++)
      sampledActions.push_back(actions[sampledIndices[t]]);
    sampledRewards = rewards.elem(sampledIndices).t();
    sampledNextStates = nextStates.cols(sampledIndices);
    isTerminal = this->isTerminal.elem(sampledIndices).t();

    // Calculate the weights of sampled transitions.

    size_t numSample = full ? capacity : position;
    weights = arma::rowvec(sampledIndices.n_rows);

    for (size_t i = 0; i < sampledIndices.n_rows; ++i)
    {
      double p_sample = idxSum.Get(sampledIndices(i)) / idxSum.Sum();
      weights(i) = std::pow(numSample * p_sample, -beta);
    }
    weights /= weights.max();
  }

  /**
   * Update priorities of sampled transitions.
   *
   * @param indices The indices of sample to be updated.
   * @param priorities Their corresponding priorities.
   */
  void UpdatePriorities(arma::ucolvec& indices, arma::colvec& priorities)
  {
      arma::colvec alphaPri = alpha * priorities;
      maxPriority = std::max(maxPriority, max(priorities));
      idxSum.BatchUpdate(indices, alphaPri);
  }

  /**
   * Get the number of transitions in the memory.
   *
   * @return Actual used memory size.
   */
  const size_t& Size()
  {
    return full ? capacity : position;
  }

  /**
   * Annealing the beta.
   */
  void BetaAnneal()
  {
    beta = beta + (1 - initialBeta) * 1.0 / replayBetaIters;
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
    arma::colvec tdError(target.n_cols);
    for (size_t i = 0; i < target.n_cols; i ++)
    {
      tdError(i) = nextActionValues(sampledActions[i].action, i) -
          target(sampledActions[i].action, i);
    }
    tdError = arma::abs(tdError);
    UpdatePriorities(sampledIndices, tdError);

    // Update the gradient
    gradients = arma::mean(weights) * gradients;
  }

  //! Get the number of steps for n-step agent.
  const size_t& NSteps() const { return nSteps; }

 private:
  //! Locally-stored number of examples of each sample.
  size_t batchSize;

  //! Locally-stored total memory limit.
  size_t capacity;

  //! Indicate the position to store new transition.
  size_t position;

  //! Locally-stored indicator that whether the memory is full or not.
  bool full;

  //! How much prioritization is used.
  //! (0 - no prioritization, 1 - full prioritization)
  double alpha;

  //! Locally-stored the max priority.
  double maxPriority;

  //! Initial value of beta for prioritized replay buffer.
  double initialBeta;

  //! The value of beta for current sample.
  double beta;

  //! How many iteration for replay beta to decay.
  size_t replayBetaIters;

  //! Locally-stored the prefix sum of prioritization.
  SumTree<double> idxSum;

  //! Locally-stored the indices of sampled transitions.
  arma::ucolvec sampledIndices;

  //! Locally-stored the weights of sampled transitions.
  arma::rowvec weights;

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
};

} // namespace mlpack

#endif
