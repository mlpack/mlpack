
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat.hpp:

Program Listing for File concat.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concat.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/concat.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_HPP
   #define MLPACK_METHODS_ANN_LAYER_CONCAT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "../visitor/delete_visitor.hpp"
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       typename... CustomLayers
   >
   class Concat
   {
    public:
     Concat(const bool model = false,
            const bool run = true);
   
     Concat(arma::Row<size_t>& inputSize,
            const size_t axis,
            const bool model = false,
            const bool run = true);
   
     ~Concat();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
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
     void Gradient(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& /* gradient */);
   
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
   
     std::vector<LayerTypes<CustomLayers...> >& Model()
     {
       if (model)
       {
         return network;
       }
   
       return empty;
     }
   
     const arma::mat& Parameters() const { return weights; }
     arma::mat& Parameters() { return weights; }
   
     bool Run() const { return run; }
     bool& Run() { return run; }
   
     arma::mat const& InputParameter() const { return inputParameter; }
     arma::mat& InputParameter() { return inputParameter; }
   
     arma::mat const& OutputParameter() const { return outputParameter; }
     arma::mat& OutputParameter() { return outputParameter; }
   
     arma::mat const& Delta() const { return delta; }
     arma::mat& Delta() { return delta; }
   
     arma::mat const& Gradient() const { return gradient; }
     arma::mat& Gradient() { return gradient; }
   
     size_t const& ConcatAxis() const { return axis; }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar,  const uint32_t /* version */);
   
    private:
     arma::Row<size_t> inputSize;
   
     size_t axis;
   
     bool useAxis;
   
     bool model;
   
     bool run;
   
     size_t channels;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     OutputDataType weights;
   
     DeltaVisitor deltaVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     DeleteVisitor deleteVisitor;
   
     std::vector<LayerTypes<CustomLayers...> > empty;
   
     arma::mat delta;
   
     arma::mat inputParameter;
   
     arma::mat outputParameter;
   
     arma::mat gradient;
   }; // class Concat
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "concat_impl.hpp"
   
   #endif
