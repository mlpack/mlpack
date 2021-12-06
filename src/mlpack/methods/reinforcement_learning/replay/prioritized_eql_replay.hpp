/**
 * @file methods/reinforcement_learning/replay/prioritized_eql_replay.hpp
 * @author Nanubala Gnana Sai
 *
 * This file is an implementation of prioritized EQL experience replay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_PRIORITIZED_EQL_REPLAY_HPP
#define MLPACK_METHODS_RL_PRIORITIZED_EQL_REPLAY_HPP

#include <mlpack/prereqs.hpp>
#include "sumtree.hpp"

namespace mlpack {
namespace rl {

/**
 * Implementation of the EQL variation of prioritized experience replay. The update
 * is handled similar to Hindsight Experience Replay(HER) and the priorities are assigned
 * based on the errors.
 * 
 * For more details, see the following:
 * @code
 * @article{yang2019generalized,
 *  author    = {Yang and
 *               Runzhe and
 *               Sun and
 *               Xingyuan and
 *               Narasimhan and
 *               Karthik},
 *  title     = {A generalized algorithm for multi-objective reinforcement learning and policy adaptation},
 *  journal   = {arXiv preprint arXiv:1908.08342},
 *  year      = {2019},
 *  url       = {https://arxiv.org/abs/1908.08342}
 * }
 * @endcode
 *
 * @tparam EnvironmentType Desired task.
 */
template <typename EnvironmentType>
class PrioritizedEQLReplay
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
    arma::vec reward;
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
      replayBetaIters(0)
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of prioritized experience replay class.
   *
   * @param batchSize Number of examples returned at each sample.
   * @param capacity Total memory size in terms of number of examples.
   * @param alpha How much prioritization is used.
   * @param dimension The dimension of an encoded state.
   */
  PrioritizedEQLReplay(const size_t batchSize,
                       const size_t capacity,
                       const double alpha,
                       const size_t dimension = StateType::dimension) :
      batchSize(batchSize),
      capacity(capacity),
      position(0),
      full(false),
      alpha(alpha),
      maxPriority(1.0),
      initialBeta(0.6),
      replayBetaIters(10000),
      states(dimension, capacity),
      actions(capacity),
      rewards(capacity),
      nextStates(dimension, capacity),
      preferences(EnvironmentType::rewardSize, capacity),
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
   * @param preference Given preference.
   * @param isEnd Whether next state is terminal state.
   * @param discount The discount parameter.
   */
  void Store(StateType state,
             ActionType action,
             double reward,
             StateType nextState,
             PreferenceType preference,
             bool isEnd,
             const double& discount)
  {
    states.col(position) = state.Encode();
    actions[position] = action;
    rewards(position) = reward;
    nextStates.col(position) = nextState.Encode();
    preferences.col(position) = preference;
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
   * Sample some experience according to their priorities.
   *
   * @return The indices to be chosen.
   */
  arma::uvec SampleProportional()
  {
    arma::uvec idxes(batchSize);
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
  void Sample(arma::mat& /* sampledStates */,
              std::vector<ActionType>& /* sampledActions */,
              arma::rowvec& /* sampledRewards */,
              arma::mat& /* sampledNextStates */,
              arma::irowvec& /* isTerminal */)
  { /* Nothing to do here */ }

  /**
   * Sample some experience according to their priorities.
   *
   * @param sampledStates Sampled encoded states.
   * @param sampledActions Sampled actions.
   * @param sampledRewards Sampled rewards.
   * @param sampledNextStates Sampled encoded next states.
   * @param sampledPreferences Sampled preferences.
   * @param isTerminal Indicate whether corresponding next state is terminal
   *        state.
   */
  void SampleEQL(arma::mat& sampledStates,
                 std::vector<ActionType>& sampledActions,
                 arma::rowvec& sampledRewards,
                 arma::mat& sampledNextStates,
                 arma::mat& samplePreferences,
                 arma::irowvec& isTerminal)
  {
    sampledIndices = SampleProportional();
    BetaAnneal();

    sampledStates = states.cols(sampledIndices);
    for (size_t t = 0; t < sampledIndices.n_rows; t ++)
      sampledActions.push_back(actions[sampledIndices[t]]);
    sampledRewards = rewards.elem(sampledIndices).t();
    sampledNextStates = nextStates.cols(sampledIndices);
    sampledPreferences = preferences.cols(sampledIndices);
    isTerminal = this->isTerminal.elem(sampledIndices).t();

    // Calculate the weights of sampled transitions.
    size_t numSample = full ? capacity : position;
    weights = arma::rowvec(sampledIndices.n_rows);

    for (size_t i = 0; i < sampledIndices.n_rows; ++i)
    {
      double p_sample = idxSum.Get(sampledIndices(i)) / idxSum.Sum();
      weights(i) = pow(numSample * p_sample, -beta);
    }
    weights /= weights.max();
  }

  /**
   * Update priorities of sampled transitions.
   *
   * @param indices The indices of sample to be updated.
   * @param priorities Their corresponding priorities.
   */
  void UpdatePriorities(arma::uvec& indices, arma::vec& priorities)
  {
      arma::vec alphaPri = alpha * priorities;
      maxPriority = std::max(maxPriority, arma::max(priorities));
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
   * Update the priorities of transitions and the gradients.
   *
   * @param learningActionValue Sampled statePref's learningNetwork action-value.
   * @param sampledActions Agent's sampled action.
   * @param nextLearningActionValue Sampled next statePref's learningNetwork action-value.
   * @param sampledPreferences Agent's current preference.
   * @param discount The current discount factor.
   * @param gradients The model's gradients.
   */
  void Update(const arma::mat& learningActionValue,
              const std::vector<ActionType>& sampledActions,
              const arma::mat& nextLearningActionValue,
              const arma::mat& sampledPreferences,
              const double discount,
              arma::mat& gradients)
  {
    size_t inputSize = sampledActions.size();
    arma::vec tdError(inputSize);
    arma::cube actionValCube(learningActionValue.memptr(), EnvironmentType::rewardSize,
                             ActionType::size, inputSize);
    arma::cube nextActionValCube(nextLearningActionValue.memptr(), EnvironmentType::rewardSize,
                                 ActionType::size, inputSize);
    for (size_t i = 0; i < inputSize; ++i)
    {
      double wq = arma::dot(sampledPreferences.col(i), actionValCube.slice(i).col(sampledActions[i].action));
      double wr = arma::dot(sampledPreferences.col(i), sampledRewards.col(i));
      size_t bestAction = arma::max_index(nextActionValCube.slice(i).each_col() % sampledPreferences.col(i));
      double whq = arma::dot(sampledPreferences.col(i), nextActionValueCube.slice(i).col(bestAction));

      tdError(i) = arma::abs(wr - wq + discount * whq * (1 - isTerminal(i)));
    }

    tdError += epsilon;
    UpdatePriorities(sampledIndices, tdError);

    // Update the gradient
    gradients = arma::mean(weights) * gradients;
  }

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
  arma::uvec sampledIndices;

  //! Locally-stored the weights of sampled transitions.
  arma::rowvec weights;

  //! Locally-stored encoded previous states.
  arma::mat states;

  //! Locally-stored previous actions.
  std::vector<ActionType> actions;

  //! Locally-stored previous rewards.
  arma::mat rewardLists;

  //! Locally-stored encoded previous next states.
  arma::mat nextStates;

  //! Locally-stored encoded previous preferenecs.
  arma::mat preferences;

  //! Locally-stored termination information of previous experience.
  arma::irowvec isTerminal;

  //! Locally stored reward size for convenience.
  const size_t rewardSize = EnvironmentType::rewardSize;
};

} // namespace rl
} // namespace mlpack

#endif
