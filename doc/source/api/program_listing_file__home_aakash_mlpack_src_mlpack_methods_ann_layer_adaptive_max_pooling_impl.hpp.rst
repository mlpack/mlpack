
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_max_pooling_impl.hpp:

Program Listing for File adaptive_max_pooling_impl.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_adaptive_max_pooling_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/adaptive_max_pooling_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_ADAPTIVE_MAX_POOLING_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "adaptive_max_pooling.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   AdaptiveMaxPooling<InputDataType, OutputDataType>::AdaptiveMaxPooling()
   {
     // Nothing to do here.
   }
   
   template <typename InputDataType, typename OutputDataType>
   AdaptiveMaxPooling<InputDataType, OutputDataType>::AdaptiveMaxPooling(
       const size_t outputWidth,
       const size_t outputHeight) :
       AdaptiveMaxPooling(std::tuple<size_t, size_t>(outputWidth, outputHeight))
   {
     // Nothing to do here.
   }
   
   template <typename InputDataType, typename OutputDataType>
   AdaptiveMaxPooling<InputDataType, OutputDataType>::AdaptiveMaxPooling(
       const std::tuple<size_t, size_t>& outputShape):
       outputWidth(std::get<0>(outputShape)),
       outputHeight(std::get<1>(outputShape)),
       reset(false)
   {
     poolingLayer = ann::MaxPooling<>(0, 0);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void AdaptiveMaxPooling<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     if (!reset)
     {
       IntializeAdaptivePadding();
       reset = true;
     }
   
     poolingLayer.Forward(input, output);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void AdaptiveMaxPooling<InputDataType, OutputDataType>::Backward(
     const arma::Mat<eT>& input,
     const arma::Mat<eT>& gy,
     arma::Mat<eT>& g)
   {
     poolingLayer.Backward(input, gy, g);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void AdaptiveMaxPooling<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(outputWidth));
     ar(CEREAL_NVP(outputHeight));
     ar(CEREAL_NVP(reset));
     ar(CEREAL_NVP(poolingLayer));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
