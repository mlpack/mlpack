
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_merge.hpp:

Program Listing for File add_merge.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_merge.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/add_merge.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ADD_MERGE_HPP
   #define MLPACK_METHODS_ANN_LAYER_ADD_MERGE_HPP
   
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
   class AddMerge
   {
    public:
     AddMerge(const bool model = false, const bool run = true);
   
     AddMerge(const bool model, const bool run, const bool ownsLayers);
   
     ~AddMerge();
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& /* input */, OutputType& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g,
                   const size_t index);
   
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
      * This is the overload of Gradient() that runs a specific layer with the
      * given input.
      *
      * @param input The input parameter used for calculating the gradient.
      * @param error The calculated error.
      * @param gradient The calculated gradient.
      * @param The index of the layer to run.
      */
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient,
                   const size_t index);
   
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
   
     InputDataType const& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
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
   
     bool Run() const { return run; }
     bool& Run() { return run; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     bool model;
   
     bool run;
   
     bool ownsLayers;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     std::vector<LayerTypes<CustomLayers...> > empty;
   
     DeleteVisitor deleteVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     DeltaVisitor deltaVisitor;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   
     OutputDataType weights;
   }; // class AddMerge
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "add_merge_impl.hpp"
   
   #endif
