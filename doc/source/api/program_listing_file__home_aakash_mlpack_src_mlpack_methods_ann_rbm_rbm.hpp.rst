
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_rbm_rbm.hpp:

Program Listing for File rbm.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_rbm_rbm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/rbm/rbm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_RBM_RBM_HPP
   #define MLPACK_METHODS_ANN_RBM_RBM_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/ann/rbm/rbm_policies.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template<
     typename InitializationRuleType,
     typename DataType = arma::mat,
     typename PolicyType = BinaryRBM
   >
   class RBM
   {
    public:
     using NetworkType = RBM<InitializationRuleType, DataType, PolicyType>;
     typedef typename DataType::elem_type ElemType;
   
     RBM(arma::Mat<ElemType> predictors,
         InitializationRuleType initializeRule,
         const size_t visibleSize,
         const size_t hiddenSize,
         const size_t batchSize = 1,
         const size_t numSteps = 1,
         const size_t negSteps = 1,
         const size_t poolSize = 2,
         const ElemType slabPenalty = 8,
         const ElemType radius = 1,
         const bool persistence = false);
   
     // Reset the network.
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
     Reset();
   
     // Reset the network.
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     Reset();
   
     template<typename OptimizerType, typename... CallbackType>
     double Train(OptimizerType& optimizer, CallbackType&&... callbacks);
   
     double Evaluate(const arma::Mat<ElemType>& parameters,
                     const size_t i,
                     const size_t batchSize);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, double>::type
     FreeEnergy(const arma::Mat<ElemType>& input);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value,
         double>::type
     FreeEnergy(const arma::Mat<ElemType>& input);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
     Phase(const InputType& input, DataType& gradient);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     Phase(const InputType& input, DataType& gradient);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
     SampleHidden(const arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     SampleHidden(const arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
     SampleVisible(arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     SampleVisible(arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
     VisibleMean(InputType& input, DataType& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     VisibleMean(InputType& input, DataType& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, BinaryRBM>::value, void>::type
     HiddenMean(const InputType& input, DataType& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     HiddenMean(const InputType& input, DataType& output);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     SpikeMean(const InputType& visible, DataType& spikeMean);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     SampleSpike(InputType& spikeMean, DataType& spike);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     SlabMean(const DataType& visible, DataType& spike, DataType& slabMean);
   
     template<typename Policy = PolicyType, typename InputType = DataType>
     typename std::enable_if<std::is_same<Policy, SpikeSlabRBM>::value, void>::type
     SampleSlab(InputType& slabMean, DataType& slab);
   
     void Gibbs(const arma::Mat<ElemType>& input,
                arma::Mat<ElemType>& output,
                const size_t steps = SIZE_MAX);
   
     void Gradient(const arma::Mat<ElemType>& parameters,
                   const size_t i,
                   arma::Mat<ElemType>& gradient,
                   const size_t batchSize);
   
     void Shuffle();
   
     size_t NumFunctions() const { return numFunctions; }
   
     size_t NumSteps() const { return numSteps; }
   
     const arma::Mat<ElemType>& Parameters() const { return parameter; }
     arma::Mat<ElemType>& Parameters() { return parameter; }
   
     arma::Cube<ElemType> const& Weight() const { return weight; }
     arma::Cube<ElemType>& Weight() { return weight; }
   
     DataType const& VisibleBias() const { return visibleBias; }
     DataType& VisibleBias() { return visibleBias; }
   
     DataType const& HiddenBias() const { return hiddenBias; }
     DataType& HiddenBias() { return hiddenBias; }
   
     DataType const& SpikeBias() const { return spikeBias; }
     DataType& SpikeBias() { return spikeBias; }
   
     ElemType const& SlabPenalty() const { return 1.0 / slabPenalty; }
   
     DataType const& VisiblePenalty() const { return visiblePenalty; }
     DataType& VisiblePenalty() { return visiblePenalty; }
   
     size_t const& VisibleSize() const { return visibleSize; }
     size_t const& HiddenSize() const { return hiddenSize; }
     size_t const& PoolSize() const { return poolSize; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     arma::Mat<ElemType> parameter;
     arma::Mat<ElemType> predictors;
     // Initializer for initializing the weights of the network.
     InitializationRuleType initializeRule;
     arma::Mat<ElemType> state;
     size_t numFunctions;
     size_t visibleSize;
     size_t hiddenSize;
     size_t batchSize;
     size_t numSteps;
     size_t negSteps;
     size_t poolSize;
     size_t steps;
     arma::Cube<ElemType> weight;
     DataType visibleBias;
     DataType hiddenBias;
     DataType preActivation;
     DataType spikeBias;
     DataType visiblePenalty;
     DataType visibleMean;
     DataType spikeMean;
     DataType spikeSamples;
     DataType slabMean;
     ElemType slabPenalty;
     ElemType radius;
     arma::Mat<ElemType> hiddenReconstruction;
     arma::Mat<ElemType> visibleReconstruction;
     arma::Mat<ElemType> negativeSamples;
     arma::Mat<ElemType> negativeGradient;
     arma::Mat<ElemType> tempNegativeGradient;
     arma::Mat<ElemType> positiveGradient;
     arma::Mat<ElemType> gibbsTemporary;
     bool persistence;
     bool reset;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   #include "rbm_impl.hpp"
   #include "spike_slab_rbm_impl.hpp"
   
   #endif
