
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_spatial_dropout.hpp:

Program Listing for File spatial_dropout.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_spatial_dropout.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/spatial_dropout.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SPATIAL_DROPOUT_HPP
   #define MLPACK_METHODS_ANN_LAYER_SPATIAL_DROPOUT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class SpatialDropout
   {
    public:
     SpatialDropout();
     SpatialDropout(const size_t size, const double ratio = 0.5);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t Size() const { return size; }
   
     size_t& Size() { return size; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     double Ratio() const { return ratio; }
   
     void Ratio(const double r)
     {
       ratio = r;
       scale = 1.0 / (1.0 - ratio);
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     OutputDataType mask;
   
     size_t size;
   
     double ratio;
   
     double scale;
   
     bool reset;
   
     size_t batchSize;
   
     size_t inputSize;
   
     bool deterministic;
   }; // class SpatialDropout
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "spatial_dropout_impl.hpp"
   
   #endif
