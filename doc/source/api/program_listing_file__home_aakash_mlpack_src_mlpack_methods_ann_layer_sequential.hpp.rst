
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_sequential.hpp:

Program Listing for File sequential.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_sequential.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/sequential.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_HPP
   #define MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <boost/ptr_container/ptr_vector.hpp>
   
   #include "../visitor/delete_visitor.hpp"
   #include "../visitor/copy_visitor.hpp"
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/output_height_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   #include "../visitor/output_width_visitor.hpp"
   #include "../visitor/input_shape_visitor.hpp"
   
   #include "layer_types.hpp"
   #include "add_merge.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       bool Residual = false,
       typename... CustomLayers
   >
   class Sequential
   {
    public:
     Sequential(const bool model = true);
   
     Sequential(const bool model, const bool ownsLayers);
   
     Sequential(const Sequential& layer);
   
     Sequential& operator = (const Sequential& layer);
   
     ~Sequential();
   
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
   
     std::vector<LayerTypes<CustomLayers...> >& Model()
     {
       if (model)
       {
         return network;
       }
   
       return empty;
     }
   
     const arma::mat& Parameters() const { return parameters; }
     arma::mat& Parameters() { return parameters; }
   
     arma::mat const& InputParameter() const { return inputParameter; }
     arma::mat& InputParameter() { return inputParameter; }
   
     arma::mat const& OutputParameter() const { return outputParameter; }
     arma::mat& OutputParameter() { return outputParameter; }
   
     arma::mat const& Delta() const { return delta; }
     arma::mat& Delta() { return delta; }
   
     arma::mat const& Gradient() const { return gradient; }
     arma::mat& Gradient() { return gradient; }
   
     size_t InputShape() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     bool model;
   
     bool reset;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     arma::mat parameters;
   
     DeltaVisitor deltaVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     DeleteVisitor deleteVisitor;
   
     std::vector<LayerTypes<CustomLayers...> > empty;
   
     arma::mat delta;
   
     arma::mat inputParameter;
   
     arma::mat outputParameter;
   
     arma::mat gradient;
   
     OutputWidthVisitor outputWidthVisitor;
   
     OutputHeightVisitor outputHeightVisitor;
   
     CopyVisitor<CustomLayers...> copyVisitor;
   
     size_t width;
   
     size_t height;
   
     bool ownsLayers;
   }; // class Sequential
   
   /*
    * Convenience typedef for use as Residual<> layer.
    */
   template<
     typename InputDataType = arma::mat,
     typename OutputDataType = arma::mat,
     typename... CustomLayers
   >
   using Residual = Sequential<
       InputDataType, OutputDataType, true, CustomLayers...>;
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "sequential_impl.hpp"
   
   #endif
