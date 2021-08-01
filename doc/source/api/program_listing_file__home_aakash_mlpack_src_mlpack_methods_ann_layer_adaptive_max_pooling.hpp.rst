
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_max_pooling.hpp:

Program Listing for File adaptive_max_pooling.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_max_pooling.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/adaptive_max_pooling.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_HPP
   #define MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class AdaptiveMaxPooling
   {
    public:
     AdaptiveMaxPooling();
   
     AdaptiveMaxPooling(const size_t outputWidth,
                        const size_t outputHeight);
   
     AdaptiveMaxPooling(const std::tuple<size_t, size_t>& outputShape);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     const OutputDataType& OutputParameter() const
     { return poolingLayer.OutputParameter(); }
   
     OutputDataType& OutputParameter() { return poolingLayer.OutputParameter(); }
   
     const OutputDataType& Delta() const { return poolingLayer.Delta(); }
     OutputDataType& Delta() { return poolingLayer.Delta(); }
   
     size_t InputWidth() const { return poolingLayer.InputWidth(); }
     size_t& InputWidth() { return poolingLayer.InputWidth(); }
   
     size_t InputHeight() const { return poolingLayer.InputHeight(); }
     size_t& InputHeight() { return poolingLayer.InputHeight(); }
   
     size_t OutputWidth() const { return outputWidth; }
     size_t& OutputWidth() { return outputWidth; }
   
     size_t OutputHeight() const { return outputHeight; }
     size_t& OutputHeight() { return outputHeight; }
   
     size_t InputSize() const { return poolingLayer.InputSize(); }
   
     size_t OutputSize() const { return poolingLayer.OutputSize(); }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     void IntializeAdaptivePadding()
     {
       poolingLayer.StrideWidth() = std::floor(poolingLayer.InputWidth() /
           outputWidth);
       poolingLayer.StrideHeight() = std::floor(poolingLayer.InputHeight() /
           outputHeight);
   
       poolingLayer.KernelWidth() = poolingLayer.InputWidth() -
           (outputWidth - 1) * poolingLayer.StrideWidth();
       poolingLayer.KernelHeight() = poolingLayer.InputHeight() -
           (outputHeight - 1) * poolingLayer.StrideHeight();
   
       if (poolingLayer.KernelHeight() <= 0 || poolingLayer.KernelWidth() <= 0 ||
           poolingLayer.StrideWidth() <= 0 || poolingLayer.StrideHeight() <= 0)
       {
         Log::Fatal << "Given output shape (" << outputWidth << ", "
           << outputHeight << ") is not possible for given input shape ("
           << poolingLayer.InputWidth() << ", " << poolingLayer.InputHeight()
           << ")." << std::endl;
       }
     }
   
     MaxPooling<InputDataType, OutputDataType> poolingLayer;
   
     size_t outputWidth;
   
     size_t outputHeight;
   
     bool reset;
   }; // class AdaptiveMaxPooling
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "adaptive_max_pooling_impl.hpp"
   
   #endif
