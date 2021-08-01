
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_recurrent.hpp:

Program Listing for File recurrent.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_recurrent.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/recurrent.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_HPP
   #define MLPACK_METHODS_ANN_LAYER_RECURRENT_HPP
   
   #include <mlpack/core.hpp>
   
   #include "../visitor/delete_visitor.hpp"
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/copy_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   #include "../visitor/input_shape_visitor.hpp"
   
   #include "layer_types.hpp"
   #include "add_merge.hpp"
   #include "sequential.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       typename... CustomLayers
   >
   class Recurrent
   {
    public:
     Recurrent();
   
     Recurrent(const Recurrent&);
   
     template<typename StartModuleType,
              typename InputModuleType,
              typename FeedbackModuleType,
              typename TransferModuleType>
     Recurrent(const StartModuleType& start,
               const InputModuleType& input,
               const FeedbackModuleType& feedback,
               const TransferModuleType& transfer,
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
      * @param input The input parameter used for calculating the gradient.
      * @param error The calculated error.
      * @param gradient The calculated gradient.
      */
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& /* gradient */);
   
     std::vector<LayerTypes<CustomLayers...> >& Model() { return network; }
   
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
   
     size_t const& Rho() const { return rho; }
   
     size_t InputShape() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     DeleteVisitor deleteVisitor;
   
     CopyVisitor<CustomLayers...> copyVisitor;
   
     LayerTypes<CustomLayers...> startModule;
   
     LayerTypes<CustomLayers...> inputModule;
   
     LayerTypes<CustomLayers...> feedbackModule;
   
     LayerTypes<CustomLayers...> transferModule;
   
     size_t rho;
   
     size_t forwardStep;
   
     size_t backwardStep;
   
     size_t gradientStep;
   
     bool deterministic;
   
     bool ownsLayer;
   
     OutputDataType parameters;
   
     LayerTypes<CustomLayers...> initialModule;
   
     LayerTypes<CustomLayers...> recurrentModule;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     LayerTypes<CustomLayers...> mergeModule;
   
     DeltaVisitor deltaVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     std::vector<arma::mat> feedbackOutputParameter;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   
     arma::mat recurrentError;
   }; // class Recurrent
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "recurrent_impl.hpp"
   
   #endif
