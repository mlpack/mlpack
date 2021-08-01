
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_weight_norm.hpp:

Program Listing for File weight_norm.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_weight_norm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/weight_norm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_HPP
   #define MLPACK_METHODS_ANN_LAYER_WEIGHTNORM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "layer_types.hpp"
   
   #include "../visitor/delete_visitor.hpp"
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   #include "../visitor/reset_visitor.hpp"
   #include "../visitor/weight_size_visitor.hpp"
   #include "../visitor/weight_set_visitor.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
     typename InputDataType = arma::mat,
     typename OutputDataType = arma::mat,
     typename... CustomLayers
   >
   class WeightNorm
   {
    public:
     WeightNorm(LayerTypes<CustomLayers...> layer = LayerTypes<CustomLayers...>());
   
     ~WeightNorm();
   
     void Reset();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     LayerTypes<CustomLayers...> const& Layer() { return wrappedLayer; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t biasWeightSize;
   
     DeleteVisitor deleteVisitor;
   
     OutputDataType delta;
   
     DeltaVisitor deltaVisitor;
   
     OutputDataType gradient;
   
     LayerTypes<CustomLayers...> wrappedLayer;
   
     size_t layerWeightSize;
   
     OutputDataType outputParameter;
   
     OutputParameterVisitor outputParameterVisitor;
   
     void ResetGradients(arma::mat& gradient);
   
     ResetVisitor resetVisitor;
   
     OutputDataType scalarParameter;
   
     OutputDataType vectorParameter;
   
     OutputDataType weights;
   
     WeightSizeVisitor weightSizeVisitor;
   
     OutputDataType layerGradients;
   
     OutputDataType layerWeights;
   }; // class WeightNorm
   
   } // namespace ann
   } // namespace mlpack
   
   // Include the implementation.
   #include "weight_norm_impl.hpp"
   
   #endif
