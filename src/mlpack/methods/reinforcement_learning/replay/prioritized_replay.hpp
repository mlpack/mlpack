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
#include "random_replay.hpp"

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
class PrioritizedReplay: public RandomReplay
{
 public:
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
      max_priority(1.0)
  { 
    int size = 1;
    while (size < capacity) {
      size *= 2;
    }
    idxSum = new SumTree(size);
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

    idxSum[position] = max_priority * alpha;

    position++;
    if (position == capacity)
    {
      full = true;
      position = 0;
    }
  }

  arma::uvec sampleProportional()
  {
    arma::uvec idxes(batchSize);
    double totalSum = idxSum.sum(0, (full ? capacity : position) - 1);
    double sumPerRange = totalSum / batchSize;
    for (size_t bt = 0; bt < batchSize; bt ++) {
      double mass = arma::randu() * sumPerRange + bt * sumPerRange;
      int idx = idxSum.findPrefixSum(mass);
      idxes(bt) = idx;
    }
    return idxes;
  }

  void Sample(arma::mat& sampledStates,
              arma::icolvec& sampledActions,
              arma::colvec& sampledRewards,
              arma::mat& sampledNextStates,
              arma::icolvec& isTerminal)
  {
    size_t upperBound = full ? capacity : position;

    arma::uvec sampledIndices = sampleProportional();

    sampledStates = states.cols(sampledIndices);
    sampledActions = actions.elem(sampledIndices);
    sampledRewards = rewards.elem(sampledIndices);
    sampledNextStates = nextStates.cols(sampledIndices);
    isTerminal = this->isTerminal.elem(sampledIndices);

//    btodo: caculate the weights of sampled transitions

  }

  void update_priorities(double beta) 
  {
//    btodo: update priorities of sampled transitions.
  }

private:
  double alpha;
  double max_priority;
  SumTree idxSum;
};

} // namespace rl
} // namespace mlpack

#endif
