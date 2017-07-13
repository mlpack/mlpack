#ifndef MLPACK_METHODS_RL_TRAINING_CONFIG_HPP
#define MLPACK_METHODS_RL_TRAINING_CONFIG_HPP

namespace mlpack {
namespace rl {

class TrainingConfig
{
public:
  TrainingConfig() : stepLimit(0), gradientLimit(40), doubleQLearning(false)
  {}

  const size_t NumOfWorkers() const { return numOfWorkers; }
  size_t& NumOfWorkers() { return numOfWorkers; }

  const size_t UpdateInterval() const { return updateInterval; }
  size_t& UpdateInterval() { return updateInterval; }

  const size_t TargetNetworkSyncInterval() const { return targetNetworkSyncInterval; }
  size_t& TargetNetworkSyncInterval() { return targetNetworkSyncInterval; }

  const size_t StepLimit() const { return stepLimit; }
  size_t& StepLimit() { return stepLimit; }

  const size_t ExplorationSteps() const { return explorationSteps; }
  size_t& ExplorationSteps() { return explorationSteps; }

  const double StepSize() const { return stepSize; }
  double& StepSize() { return stepSize; }

  const double Discount() const { return discount; }
  double& Discount() { return discount; }

  const double GradientLimit() const { return gradientLimit; }
  double& GradientLimit() { return gradientLimit; }

  const bool DoubleQLearning() const { return doubleQLearning; }
  bool& DoubleQLearning() { return doubleQLearning; }

private:
  size_t numOfWorkers;
  size_t updateInterval;
  size_t targetNetworkSyncInterval;
  size_t stepLimit;
  size_t explorationSteps;
  double stepSize;
  double discount;
  double gradientLimit;
  bool doubleQLearning;
};

}
}

#endif
