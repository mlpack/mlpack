/**
 * @file training_config.hpp
 * @author Shangtong Zhang
 *
 * This file is the implementation of TrainingConfig class,
 * which contains hyper-parameters for training RL agent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_TRAINING_CONFIG_HPP
#define MLPACK_METHODS_RL_TRAINING_CONFIG_HPP

namespace mlpack {
namespace rl {

class TrainingConfig
{
public:
  TrainingConfig() : stepLimit(0), gradientLimit(40), doubleQLearning(false)
  { /* Nothing to do here. */ }

  /**
   * Get and modify the number workers.
   * This is valid only for async RL agent.
   **/
  const size_t NumOfWorkers() const { return numOfWorkers; }
  size_t& NumOfWorkers() { return numOfWorkers; }

  /**
   * Get and modify the update interval, which is similar to batch size,
   * however the update is done one by one.
   * This is valid only for async RL agent.
   */
  const size_t UpdateInterval() const { return updateInterval; }
  size_t& UpdateInterval() { return updateInterval; }

  /**
   * Get and modify the interval for syncing target network.
   * This is valid for both async RL agent and q-learning agent.
   */
  const size_t TargetNetworkSyncInterval() const { return targetNetworkSyncInterval; }
  size_t& TargetNetworkSyncInterval() { return targetNetworkSyncInterval; }

  /**
   * Get and modify the maximum steps of each episode.
   * Setting it to 0 means no limit.
   * This is valid for both async RL agent and q-learning agent.
   */
  const size_t StepLimit() const { return stepLimit; }
  size_t& StepLimit() { return stepLimit; }

  /**
   * Get and modify the exploration steps.
   * The agent won't start learn until those steps have passed.
   * This is valid only for q-learning agent.
   */
  const size_t ExplorationSteps() const { return explorationSteps; }
  size_t& ExplorationSteps() { return explorationSteps; }

  /**
   * Get and modify the step size of the optimizer.
   * This is valid for both async RL agent and q-learning agent.
   */
  const double StepSize() const { return stepSize; }
  double& StepSize() { return stepSize; }

  /**
   * Get and modify the discount rate for future reward.
   * This is valid for both async RL agent and q-learning agent.
   */
  const double Discount() const { return discount; }
  double& Discount() { return discount; }

  /**
   * Get and modify the limit of update gradient.
   * This is valid only for async RL agent.
   */
  const double GradientLimit() const { return gradientLimit; }
  double& GradientLimit() { return gradientLimit; }

  /**
   * Get and modify the indicator of double q-learning.
   * This is valid only for q-learning agent.
   */
  const bool DoubleQLearning() const { return doubleQLearning; }
  bool& DoubleQLearning() { return doubleQLearning; }

private:
  //! Locally-stored number of workers.
  size_t numOfWorkers;

  //! Locally-stored update interval.
  size_t updateInterval;

  //! Locally-stored interval for syncing target network.
  size_t targetNetworkSyncInterval;

  //! Locally-stored step limit of each episode.
  size_t stepLimit;

  //! Locally-stored exploration steps before learning.
  size_t explorationSteps;

  //! Locally-stored step size of optimizer.
  double stepSize;

  //! Locally-stored discount rate for future reward.
  double discount;

  //! Locally-stored gradient limit.
  double gradientLimit;

  //! Locally-stored indicator for double q-learning.
  bool doubleQLearning;
};

} // namespace rl
} // namespace mlpack

#endif
