/**
 * @file methods/reinforcement_learning/training_config.hpp
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

class TrainingConfig
{
 public:
  TrainingConfig() :
      numWorkers(1),
      updateInterval(1),
      targetNetworkSyncInterval(100),
      stepLimit(200),
      explorationSteps(1),
      stepSize(0.01),
      discount(0.99),
      gradientLimit(40),
      doubleQLearning(false),
      noisyQLearning(false),
      isCategorical(false),
      atomSize(51),
      vMin(0),
      vMax(200),
      rho(0.005)
  { /* Nothing to do here. */ }

  TrainingConfig(
      size_t numWorkers,
      size_t updateInterval,
      size_t targetNetworkSyncInterval,
      size_t stepLimit,
      size_t explorationSteps,
      double stepSize,
      double discount,
      double gradientLimit,
      bool doubleQLearning,
      bool noisyQLearning,
      bool isCategorical,
      size_t atomSize,
      double vMin,
      double vMax,
      double rho) :
      numWorkers(numWorkers),
      updateInterval(updateInterval),
      targetNetworkSyncInterval(targetNetworkSyncInterval),
      stepLimit(stepLimit),
      explorationSteps(explorationSteps),
      stepSize(stepSize),
      discount(discount),
      gradientLimit(gradientLimit),
      doubleQLearning(doubleQLearning),
      noisyQLearning(noisyQLearning),
      isCategorical(isCategorical),
      atomSize(atomSize),
      vMin(vMin),
      vMax(vMax),
      rho(rho)
  { /* Nothing to do here. */ }

  //! Get the amount of workers.
  size_t NumWorkers() const { return numWorkers; }
  //! Modify the amount of workers.
  size_t& NumWorkers() { return numWorkers; }

  //! Get the update interval.
  size_t UpdateInterval() const { return updateInterval; }
  //! Modify the update interval.
  size_t& UpdateInterval() { return updateInterval; }

  //! Get the interval for syncing target network.
  size_t TargetNetworkSyncInterval() const
  { return targetNetworkSyncInterval; }
  //! Modify the interval for syncing target network.
  size_t& TargetNetworkSyncInterval() { return targetNetworkSyncInterval; }

  //! Get the maximum steps of each episode.
  size_t StepLimit() const { return stepLimit; }
  /**
   * Modify the maximum steps of each episode.
   * Setting it to 0 means no limit.
   */
  size_t& StepLimit() { return stepLimit; }

  //! Get the exploration steps.
  size_t ExplorationSteps() const { return explorationSteps; }
  //! Modify the exploration steps.
  size_t& ExplorationSteps() { return explorationSteps; }

  //! Get the step size of the optimizer.
  double StepSize() const { return stepSize; }
  //! Modify the step size of the optimizer.
  double& StepSize() { return stepSize; }

  //! Get the discount rate for future reward.
  double Discount() const { return discount; }
  //! Modify the discount rate for future reward.
  double& Discount() { return discount; }

  //! Get the limit of update gradient.
  double GradientLimit() const { return gradientLimit; }
  //! Modify the limit of update gradient.
  double& GradientLimit() { return gradientLimit; }

  //! Get the indicator of double q-learning.
  bool DoubleQLearning() const { return doubleQLearning; }
  //! Modify the indicator of double q-learning.
  bool& DoubleQLearning() { return doubleQLearning; }

  //! Get the indicator of noisy q-learning.
  bool NoisyQLearning() const { return noisyQLearning; }
  //! Modify the indicator of double q-learning.
  bool& NoisyQLearning() { return noisyQLearning; }

  //! Get the indicator of categorical q-learning.
  bool IsCategorical() const { return isCategorical; }
  //! Modify the indicator of categorical q-learning.
  bool& IsCategorical() { return isCategorical; }

  //! Get the number of atoms.
  size_t AtomSize() const { return atomSize; }
  //! Modify the number of atoms.
  size_t& AtomSize() { return atomSize; }

  //! Get the minimum value for support.
  double VMin() const { return vMin; }
  //! Modify the minimum value for support.
  double& VMin() { return vMin; }

  //! Get the maximum value for support.
  double VMax() const { return vMax; }
  //! Modify the maximum value for support.
  double& VMax() { return vMax; }

  //! Get the rho value for sac.
  double Rho() const { return rho; }
  //! Modify the rho value for sac.
  double& Rho() { return rho; }

 private:
  /**
   * Locally-stored number of workers.
   * This is valid only for async RL agent.
   */
  size_t numWorkers;

  /**
   * Locally-stored update interval.
   * Update interval is similar to batch size,
   * however the update is done one by one.
   * This is valid only for async RL agent.
   */
  size_t updateInterval;

  /**
   * Locally-stored interval for syncing target network.
   * This is valid for both async RL agent and q-learning agent.
   */
  size_t targetNetworkSyncInterval;

  /**
   * Locally-stored step limit of each episode.
   * This is valid for both async RL agent and q-learning agent.
   */
  size_t stepLimit;

  /**
   * Locally-stored exploration steps before learning.
   * The agent won't start learn until those steps have passed.
   * This is valid only for q-learning agent.
   */
  size_t explorationSteps;

  /**
   * Locally-stored step size of optimizer.
   * This is valid for both async RL agent and q-learning agent.
   */
  double stepSize;

  /**
   * Locally-stored discount rate for future reward.
   * This is valid for both async RL agent and q-learning agent.
   */
  double discount;

  /**
   * Locally-stored gradient limit.
   * This is valid only for async RL agent.
   */
  double gradientLimit;

  /**
   * Locally-stored indicator for double q-learning.
   * This is valid only for q-learning agent.
   */
  bool doubleQLearning;

  /**
   * Locally-stored indicator for noisy q-learning.
   * This is valid only for q-learning agent.
   */
  bool noisyQLearning;

  /**
   * Locally-stored indicator for categorical q-learning.
   * This is valid only for q-learning agent.
   */
  bool isCategorical;

  /**
   * Locally-stored number of atoms to be used.
   * This is valid only for categorical q-network.
   */
  size_t atomSize;

  /**
   * Locally-stored minimum value of support.
   * This is valid only for categorical q-network.
   */
  double vMin;

  /**
   * Locally-stored maximum value of support.
   * This is valid only for categorical q-network.
   */
  double vMax;

  /**
   * Locally-stored parameter for softly updating q networks.
   * This is valid only for Soft Actor-Critic.
   */
  double rho;
};

} // namespace mlpack

#endif
