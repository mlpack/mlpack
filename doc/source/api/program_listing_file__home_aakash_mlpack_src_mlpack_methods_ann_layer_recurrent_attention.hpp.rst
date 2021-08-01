
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_recurrent_attention.hpp:

Program Listing for File recurrent_attention.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_recurrent_attention.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/recurrent_attention.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_HPP
   #define MLPACK_METHODS_ANN_LAYER_RECURRENT_ATTENTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <boost/ptr_container/ptr_vector.hpp>
   
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   #include "../visitor/reset_visitor.hpp"
   #include "../visitor/weight_size_visitor.hpp"
   
   #include "layer_types.hpp"
   #include "add_merge.hpp"
   #include "sequential.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class RecurrentAttention
   {
    public:
     RecurrentAttention();
   
     template<typename RNNModuleType, typename ActionModuleType>
     RecurrentAttention(const size_t outSize,
                        const RNNModuleType& rnn,
                        const ActionModuleType& action,
                        const size_t rho);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     /*
      * Calculate the gradient using the output delta and the input activation.
      *
      * @param * (input) The input parameter used for calculating the gradient.
      * @param * (error) The calculated error.
      * @param * (gradient) The calculated gradient.
      */
     template<typename eT>
     void Gradient(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& /* error */,
                   arma::Mat<eT>& /* gradient */);
   
     std::vector<LayerTypes<>>& Model() { return network; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     OutputDataType const& Parameters() const { return parameters; }
     OutputDataType& Parameters() { return parameters; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     size_t OutSize() const { return outSize; }
   
     size_t const& Rho() const { return rho; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     void IntermediateGradient()
     {
       intermediateGradient.zeros();
   
       // Gradient of the action module.
       if (backwardStep == (rho - 1))
       {
         boost::apply_visitor(GradientVisitor(initialInput, actionError),
             actionModule);
       }
       else
       {
         boost::apply_visitor(GradientVisitor(boost::apply_visitor(
             outputParameterVisitor, actionModule), actionError),
             actionModule);
       }
   
       // Gradient of the recurrent module.
       boost::apply_visitor(GradientVisitor(boost::apply_visitor(
           outputParameterVisitor, rnnModule), recurrentError),
           rnnModule);
   
       attentionGradient += intermediateGradient;
     }
   
     size_t outSize;
   
     LayerTypes<> rnnModule;
   
     LayerTypes<> actionModule;
   
     size_t rho;
   
     size_t forwardStep;
   
     size_t backwardStep;
   
     bool deterministic;
   
     OutputDataType parameters;
   
     std::vector<LayerTypes<>> network;
   
     WeightSizeVisitor weightSizeVisitor;
   
     DeltaVisitor deltaVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     std::vector<arma::mat> feedbackOutputParameter;
   
     std::vector<arma::mat> moduleOutputParameter;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   
     arma::mat recurrentError;
   
     arma::mat actionError;
   
     arma::mat actionDelta;
   
     arma::mat rnnDelta;
   
     arma::mat initialInput;
   
     ResetVisitor resetVisitor;
   
     arma::mat attentionGradient;
   
     arma::mat intermediateGradient;
   }; // class RecurrentAttention
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "recurrent_attention_impl.hpp"
   
   #endif
