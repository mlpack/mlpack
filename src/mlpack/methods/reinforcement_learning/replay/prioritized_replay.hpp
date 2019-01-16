/**
 * @file prioritized_experience_replay.hpp
 * @author Xiaohong
 *
 * This file is an implementation of prioritized experience repla y.
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
namespace rl {

/**
 * Implementation of prioritized experience replay.
 *
 *
 * @code
 * @article{schaul2015prioritized,
 *  title={Prioritized experience replay},
 *  author={Schaul, Tom and Quan, John and Antonoglou, Ioannis and Silver, David},
 *  journal={arXiv preprint arXiv:1511.05952},
 *  year={2015}
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

  PrioritizedReplay()
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of prioritized experience replay class.
   *
   * @param batchSize Number of examples returned at each sample.
   * @param capacity Total memory size in terms of number of examples.
   * @param alpha
   * @param dimension The dimension of an encoded state.
   */
  PrioritizedReplay(const size_t batchSize,
               const size_t capacity,
               const double alpha,
               const size_t dimension = StateType::dimension) :
      batchSize(batchSize),
      capacity(capacity),
      alpha(alpha),
      position(0),
      states(dimension, capacity),
      actions(capacity),
      rewards(capacity),
      nextStates(dimension, capacity),
      isTerminal(capacity),
      full(false),
      max_priority(1.0),
      initial_beta(0.6),
      replay_beta_iters(10000)
  { 
    int size = 1;
    while (size < capacity) {
      size *= 2;
    }

    beta = initial_beta;
    idxSum = SumTree<double>(size);
  }

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

    idxSum.set(position, max_priority * alpha);

    position++;
    if (position == capacity)
    {
      full = true;
      position = 0;
    }
  }

  arma::ucolvec sampleProportional()
  {
    arma::ucolvec idxes(batchSize);
    double totalSum = idxSum.sum(0, (full ? capacity : position) - 1);
    double sumPerRange = totalSum / batchSize;
    for (size_t bt = 0; bt < batchSize; bt ++) {
      double mass = arma::randu() * sumPerRange + bt * sumPerRange;
      size_t idx = idxSum.findPrefixSum(mass);
      idxes[bt] = idx;
    }
    return idxes;
  }

  void Sample(arma::mat& sampledStates,
              arma::icolvec& sampledActions,
              arma::colvec& sampledRewards,
              arma::mat& sampledNextStates,
              arma::icolvec& isTerminal,
              arma::ucolvec& sampledIndices,
              arma::rowvec& weights)
  {
    size_t upperBound = full ? capacity : position;

    sampledIndices = sampleProportional();

    sampledStates = states.cols(sampledIndices);
    sampledActions = actions.elem(sampledIndices);
    sampledRewards = rewards.elem(sampledIndices);
    sampledNextStates = nextStates.cols(sampledIndices);
    isTerminal = this->isTerminal.elem(sampledIndices);

//  calculate the weights of sampled transitions

    size_t num_sample = full ? capacity : position;

    for (size_t i = 0; i < sampledIndices.n_rows; ++ i)
    {
      double p_sample = idxSum.get(sampledIndices[i]) / idxSum.sum();
      weights[i] = pow(num_sample * p_sample, -beta);
    }
    weights /= weights.max();
  }

  void update_priorities(arma::ucolvec& indices, arma::colvec& priorities)
  {
//    update priorities of sampled transitions.
      for (size_t i = 0; i < indices.n_rows; ++i)
      {
        idxSum.set(indices[i], alpha * priorities[i]);
        max_priority = std::max(max_priority, priorities[i]);
      }
  }

  void betaAnneal()
  {
    beta = beta + (1 - initial_beta) * 1.0 / replay_beta_iters;
  }

private:
  //! How much prioritization is used.
  double alpha;

  double max_priority;

  double initial_beta;

  double beta;

  size_t replay_beta_iters;

  //! Locally-stored the prefix sum of prioritization
  SumTree<double> idxSum;

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

};

} // namespace rl
} // namespace mlpack

#endif
