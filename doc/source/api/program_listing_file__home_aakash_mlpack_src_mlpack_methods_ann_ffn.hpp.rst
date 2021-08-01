
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_ffn.hpp:

Program Listing for File ffn.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_ffn.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/ffn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_FFN_HPP
   #define MLPACK_METHODS_ANN_FFN_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "visitor/delete_visitor.hpp"
   #include "visitor/delta_visitor.hpp"
   #include "visitor/output_height_visitor.hpp"
   #include "visitor/output_parameter_visitor.hpp"
   #include "visitor/output_width_visitor.hpp"
   #include "visitor/reset_visitor.hpp"
   #include "visitor/weight_size_visitor.hpp"
   #include "visitor/copy_visitor.hpp"
   #include "visitor/loss_visitor.hpp"
   
   #include "init_rules/network_init.hpp"
   
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <ensmallen.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template<
     typename OutputLayerType = NegativeLogLikelihood<>,
     typename InitializationRuleType = RandomInitialization,
     typename... CustomLayers
   >
   class FFN
   {
    public:
     using NetworkType = FFN<OutputLayerType, InitializationRuleType>;
   
     FFN(OutputLayerType outputLayer = OutputLayerType(),
         InitializationRuleType initializeRule = InitializationRuleType());
   
     FFN(const FFN&);
   
     FFN(FFN&&);
   
     FFN& operator = (FFN);
   
     ~FFN();
   
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
     double Train(arma::mat predictors,
                  arma::mat responses,
                  OptimizerType& optimizer,
                  CallbackTypes&&... callbacks);
   
     template<typename OptimizerType = ens::RMSProp, typename... CallbackTypes>
     double Train(arma::mat predictors,
                  arma::mat responses,
                  CallbackTypes&&... callbacks);
   
     void Predict(arma::mat predictors, arma::mat& results);
   
     template<typename PredictorsType, typename ResponsesType>
     double Evaluate(const PredictorsType& predictors,
                     const ResponsesType& responses);
   
     double Evaluate(const arma::mat& parameters);
   
     double Evaluate(const arma::mat& parameters,
                     const size_t begin,
                     const size_t batchSize,
                     const bool deterministic);
   
     double Evaluate(const arma::mat& parameters,
                     const size_t begin,
                     const size_t batchSize);
   
     template<typename GradType>
     double EvaluateWithGradient(const arma::mat& parameters, GradType& gradient);
   
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
   
     const std::vector<LayerTypes<CustomLayers...> >& Model() const
     {
       return network;
     }
     std::vector<LayerTypes<CustomLayers...> >& Model() { return network; }
   
     size_t NumFunctions() const { return numFunctions; }
   
     const arma::mat& Parameters() const { return parameter; }
     arma::mat& Parameters() { return parameter; }
   
     const arma::mat& Responses() const { return responses; }
     arma::mat& Responses() { return responses; }
   
     const arma::mat& Predictors() const { return predictors; }
     arma::mat& Predictors() { return predictors; }
   
     void ResetParameters();
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     template<typename PredictorsType, typename ResponsesType>
     void Forward(const PredictorsType& inputs, ResponsesType& results);
   
     template<typename PredictorsType, typename ResponsesType>
     void Forward(const PredictorsType& inputs ,
                  ResponsesType& results,
                  const size_t begin,
                  const size_t end);
   
     template<typename PredictorsType,
              typename TargetsType,
              typename GradientsType>
     double Backward(const PredictorsType& inputs,
                     const TargetsType& targets,
                     GradientsType& gradients);
   
    private:
     // Helper functions.
     template<typename InputType>
     void Forward(const InputType& input);
   
     void ResetData(arma::mat predictors, arma::mat responses);
   
     void Backward();
   
     template<typename InputType>
     void Gradient(const InputType& input);
   
     void ResetDeterministic();
   
     void ResetGradients(arma::mat& gradient);
   
     void Swap(FFN& network);
   
     OutputLayerType outputLayer;
   
     InitializationRuleType initializeRule;
   
     size_t width;
   
     size_t height;
   
     bool reset;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     arma::mat predictors;
   
     arma::mat responses;
   
     arma::mat parameter;
   
     size_t numFunctions;
   
     arma::mat error;
   
     DeltaVisitor deltaVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     WeightSizeVisitor weightSizeVisitor;
   
     OutputWidthVisitor outputWidthVisitor;
   
     OutputHeightVisitor outputHeightVisitor;
   
     LossVisitor lossVisitor;
   
     ResetVisitor resetVisitor;
   
     DeleteVisitor deleteVisitor;
   
     bool deterministic;
   
     arma::mat delta;
   
     arma::mat inputParameter;
   
     arma::mat outputParameter;
   
     arma::mat gradient;
   
     CopyVisitor<CustomLayers...> copyVisitor;
   
     // The GAN class should have access to internal members.
     template<
       typename Model,
       typename InitializerType,
       typename NoiseType,
       typename PolicyType
     >
     friend class GAN;
   }; // class FFN
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "ffn_impl.hpp"
   
   #endif
