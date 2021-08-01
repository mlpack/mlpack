
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_training_config.hpp:

Program Listing for File training_config.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_training_config.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/training_config.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_TRAINING_CONFIG_HPP
   #define MLPACK_METHODS_RL_TRAINING_CONFIG_HPP
   
   namespace mlpack {
   namespace rl {
   
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
   
     size_t NumWorkers() const { return numWorkers; }
     size_t& NumWorkers() { return numWorkers; }
   
     size_t UpdateInterval() const { return updateInterval; }
     size_t& UpdateInterval() { return updateInterval; }
   
     size_t TargetNetworkSyncInterval() const
     { return targetNetworkSyncInterval; }
     size_t& TargetNetworkSyncInterval() { return targetNetworkSyncInterval; }
   
     size_t StepLimit() const { return stepLimit; }
     size_t& StepLimit() { return stepLimit; }
   
     size_t ExplorationSteps() const { return explorationSteps; }
     size_t& ExplorationSteps() { return explorationSteps; }
   
     double StepSize() const { return stepSize; }
     double& StepSize() { return stepSize; }
   
     double Discount() const { return discount; }
     double& Discount() { return discount; }
   
     double GradientLimit() const { return gradientLimit; }
     double& GradientLimit() { return gradientLimit; }
   
     bool DoubleQLearning() const { return doubleQLearning; }
     bool& DoubleQLearning() { return doubleQLearning; }
   
     bool NoisyQLearning() const { return noisyQLearning; }
     bool& NoisyQLearning() { return noisyQLearning; }
   
     bool IsCategorical() const { return isCategorical; }
     bool& IsCategorical() { return isCategorical; }
   
     size_t AtomSize() const { return atomSize; }
     size_t& AtomSize() { return atomSize; }
   
     double VMin() const { return vMin; }
     double& VMin() { return vMin; }
   
     double VMax() const { return vMax; }
     double& VMax() { return vMax; }
   
     double Rho() const { return rho; }
     double& Rho() { return rho; }
   
    private:
     size_t numWorkers;
   
     size_t updateInterval;
   
     size_t targetNetworkSyncInterval;
   
     size_t stepLimit;
   
     size_t explorationSteps;
   
     double stepSize;
   
     double discount;
   
     double gradientLimit;
   
     bool doubleQLearning;
   
     bool noisyQLearning;
   
     bool isCategorical;
   
     size_t atomSize;
   
     double vMin;
   
     double vMax;
   
     double rho;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
