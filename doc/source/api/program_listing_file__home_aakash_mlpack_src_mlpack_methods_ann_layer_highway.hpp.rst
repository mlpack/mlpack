
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_highway.hpp:

Program Listing for File highway.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_highway.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/highway.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_HIGHWAY_HPP
   #define MLPACK_METHODS_ANN_LAYER_HIGHWAY_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "../visitor/delete_visitor.hpp"
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/output_height_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   #include "../visitor/output_width_visitor.hpp"
   
   #include "layer_types.hpp"
   #include "add_merge.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       typename... CustomLayers>
   class Highway
   {
    public:
     Highway();
   
     Highway(const size_t inSize, const bool model = true);
   
     ~Highway();
   
     void Reset();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     template <class LayerType, class... Args>
     void Add(Args... args)
     {
       network.push_back(new LayerType(args...));
       networkOwnerships.push_back(true);
     }
   
     void Add(LayerTypes<CustomLayers...> layer)
     {
       network.push_back(layer);
       networkOwnerships.push_back(false);
     }
   
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
   
     InputDataType const& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     size_t InSize() const { return inSize; }
   
     size_t InputShape() const
     {
       return inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inSize;
   
     bool model;
   
     bool reset;
   
     std::vector<LayerTypes<CustomLayers...> > network;
   
     std::vector<bool> networkOwnerships;
   
     std::vector<LayerTypes<CustomLayers...> > empty;
   
     OutputDataType weights;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType transformWeight;
   
     OutputDataType transformBias;
   
     OutputDataType transformGate;
   
     OutputDataType transformGateActivation;
   
     OutputDataType transformGateError;
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   
     size_t width;
   
     size_t height;
   
     OutputDataType networkOutput;
   
     DeltaVisitor deltaVisitor;
   
     OutputParameterVisitor outputParameterVisitor;
   
     DeleteVisitor deleteVisitor;
   
     OutputWidthVisitor outputWidthVisitor;
   
     OutputHeightVisitor outputHeightVisitor;
   }; // class Highway
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "highway_impl.hpp"
   
   #endif
