
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_rnn.hpp:

Program Listing for File rnn.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_rnn.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/rnn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_RNN_HPP
   #define MLPACK_METHODS_ANN_RNN_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "visitor/delete_visitor.hpp"
   #include "visitor/delta_visitor.hpp"
   #include "visitor/output_parameter_visitor.hpp"
   #include "visitor/reset_visitor.hpp"
   
   #include "init_rules/network_init.hpp"
   
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   
   #include <ensmallen.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template<
     typename OutputLayerType = NegativeLogLikelihood<>,
     typename InitializationRuleType = RandomInitialization,
     typename... CustomLayers
   >
   class RNN
   {
    public:
     using NetworkType = RNN<OutputLayerType,
                             InitializationRuleType,
                             CustomLayers...>;
   
     RNN(const size_t rho,
         const bool single = false,
         OutputLayerType outputLayer = OutputLayerType(),
         InitializationRuleType initializeRule = InitializationRuleType());
   
     RNN(const RNN&);
   
     RNN(RNN&&);
   
     RNN& operator=(const RNN&);
   
     RNN& operator=(RNN&&);
   
     ~RNN();
   
     template<typename OptimizerType>
     typename std::enable_if<
         HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
         ::value, void>::type
     WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;
   
     template<typename OptimizerType>
     typename std::enable_if<
         !HasMaxIterations<OptimizerType, size_t&(OptimizerType::*)()>
         ::value, void>::type
     WarnMessageMaxIterations(OptimizerType& optimizer, size_t samples) const;
   
     template<typename OptimizerType, typename... CallbackTypes>
     double Train(arma::cube predictors,
                  arma::cube responses,
                  OptimizerType& optimizer,
                  CallbackTypes&&... callbacks);
   
     template<typename OptimizerType = ens::StandardSGD, typename... CallbackTypes>
     double Train(arma::cube predictors,
                  arma::cube responses,
                  CallbackTypes&&... callbacks);
   
     void Predict(arma::cube predictors,
                  arma::cube& results,
                  const size_t batchSize = 256);
   
     double Evaluate(const arma::mat& parameters,
                     const size_t begin,
                     const size_t batchSize,
                     const bool deterministic);
   
     double Evaluate(const arma::mat& parameters,
                     const size_t begin,
                     const size_t batchSize);
   
     template<typename GradType>
     double EvaluateWithGradient(const arma::mat& parameters,
                                 const size_t begin,
                                 GradType& gradient,
                                 const size_t batchSize);
   
     void Gradient(const arma::mat& parameters,
                   const size_t begin,
                   arma::mat& gradient,
                   const size_t batchSize);
   
     void Shuffle();
   
     /*
      * Add a new module to the model.
      *
      * @param args The layer parameter.
      */
     template <class LayerType, class... Args>
     void Add(Args... args) { network.push_back(new LayerType(args...)); }
   
     /*
      * Add a new module to the model.
      *
      * @param layer The Layer to be added to the model.
      */
     void Add(LayerTypes<CustomLayers...> layer) { network.push_back(layer); }
   
     size_t NumFunctions() const { return numFunctions; }
   
     const arma::mat& Parameters() const { return parameter; }
     arma::mat& Parameters() { return parameter; }
   
     const size_t& Rho() const { return rho; }
     size_t& Rho() { return rho; }
   
     const arma::cube& Responses() const { return responses; }
     arma::cube& Responses() { return responses; }
   
     const arma::cube& Predictors() const { return predictors; }
     arma::cube& Predictors() { return predictors; }
   
     void Reset();
   
     void ResetParameters();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     // Helper functions.
     template<typename InputType>
     void Forward(const InputType& input);
   
     void ResetCells();
   
     void Backward();
   
     template<typename InputType>
     void Gradient(const InputType& input);
   
     void ResetDeterministic();
   
     void ResetGradients(arma::mat& gradient);
   
     size_t rho;
   
     OutputLayerType outputLayer;
   
     InitializationRuleType initializeRule;
   
     size_t inputSize;
   
     size_t outputSize;
   
     size_t targetSize;
   
     bool reset;
   
     bool single;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     arma::cube predictors;
   
     arma::cube responses;
   
     arma::mat parameter;
   
     size_t numFunctions;
   
     arma::mat error;
   
     DeltaVisitor deltaVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     std::vector<arma::mat> moduleOutputParameter;
   
     WeightSizeVisitor weightSizeVisitor;
   
     CopyVisitor<CustomLayers...> copyVisitor;
   
     ResetVisitor resetVisitor;
   
     DeleteVisitor deleteVisitor;
   
     bool deterministic;
   
     arma::mat currentGradient;
   
     // The BRN class should have access to internal members.
     template<
       typename OutputLayerType1,
       typename MergeLayerType1,
       typename MergeOutputType1,
       typename InitializationRuleType1,
       typename... CustomLayers1
     >
     friend class BRNN;
   }; // class RNN
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "rnn_impl.hpp"
   
   #endif
