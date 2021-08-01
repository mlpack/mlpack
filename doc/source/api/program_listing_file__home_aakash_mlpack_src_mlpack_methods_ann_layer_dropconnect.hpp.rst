
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_dropconnect.hpp:

Program Listing for File dropconnect.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_dropconnect.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/dropconnect.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_DROPCONNECT_HPP
   #define MLPACK_METHODS_ANN_LAYER_DROPCONNECT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "layer_types.hpp"
   #include "add_merge.hpp"
   #include "linear.hpp"
   #include "sequential.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class DropConnect
   {
    public:
     DropConnect();
   
     DropConnect(const size_t inSize,
                 const size_t outSize,
                 const double ratio = 0.5);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& /* gradient */);
   
     std::vector<LayerTypes<> >& Model() { return network; }
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     bool Deterministic() const { return deterministic; }
   
     bool &Deterministic() { return deterministic; }
   
     double Ratio() const { return ratio; }
   
     void Ratio(const double r)
     {
       ratio = r;
       scale = 1.0 / (1.0 - ratio);
     }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     double ratio;
   
     double scale;
   
     OutputDataType weights;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   
     OutputDataType mask;
   
     bool deterministic;
   
     OutputDataType denoise;
   
     LayerTypes<> baseLayer;
   
     std::vector<LayerTypes<> > network;
   }; // class DropConnect.
   
   }  // namespace ann
   }  // namespace mlpack
   
   // Include implementation.
   #include "dropconnect_impl.hpp"
   
   #endif
