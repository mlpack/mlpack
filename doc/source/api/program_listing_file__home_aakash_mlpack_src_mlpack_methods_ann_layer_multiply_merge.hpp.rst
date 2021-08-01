
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_merge.hpp:

Program Listing for File multiply_merge.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_merge.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/multiply_merge.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_HPP
   #define MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "../visitor/delete_visitor.hpp"
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       typename... CustomLayers
   >
   class MultiplyMerge
   {
    public:
     MultiplyMerge(const bool model = false, const bool run = true);
   
     MultiplyMerge(const MultiplyMerge& layer);
   
     MultiplyMerge(MultiplyMerge&& layer);
   
     MultiplyMerge& operator=(const MultiplyMerge& layer);
   
     MultiplyMerge& operator=(MultiplyMerge&& layer);
   
     ~MultiplyMerge();
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& /* input */, OutputType& output);
   
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
                   arma::Mat<eT>& gradient);
   
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
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     std::vector<LayerTypes<CustomLayers...> >& Model()
     {
       if (model)
       {
         return network;
       }
   
       return empty;
     }
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     bool model;
   
     bool run;
   
     bool ownsLayer;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     std::vector<LayerTypes<CustomLayers...> > empty;
   
     DeleteVisitor deleteVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     DeltaVisitor deltaVisitor;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   
     OutputDataType weights;
   }; // class MultiplyMerge
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "multiply_merge_impl.hpp"
   
   #endif
