
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_dropout_impl.hpp:

Program Listing for File dropout_impl.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_dropout_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/dropout_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_DROPOUT_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_DROPOUT_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "dropout.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Dropout<InputDataType, OutputDataType>::Dropout(
       const double ratio) :
       ratio(ratio),
       scale(1.0 / (1.0 - ratio)),
       deterministic(false)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   Dropout<InputDataType, OutputDataType>::Dropout(
       const Dropout& layer) :
       ratio(layer.ratio),
       scale(layer.scale),
       deterministic(layer.deterministic)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   Dropout<InputDataType, OutputDataType>::Dropout(
       const Dropout&& layer) :
       ratio(std::move(layer.ratio)),
       scale(std::move(scale)),
       deterministic(std::move(deterministic))
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   Dropout<InputDataType, OutputDataType>&
   Dropout<InputDataType, OutputDataType>::
   operator=(const Dropout& layer)
   {
     if (this != &layer)
     {
       ratio = layer.ratio;
       scale = layer.scale;
       deterministic = layer.deterministic;
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType>
   Dropout<InputDataType, OutputDataType>&
   Dropout<InputDataType, OutputDataType>::
   operator=(Dropout&& layer)
   {
     if (this != &layer)
     {
       ratio = std::move(layer.ratio);
       scale = std::move(layer.scale);
       deterministic = std::move(layer.deterministic);
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Dropout<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input,
       arma::Mat<eT>& output)
   {
     // The dropout mask will not be multiplied in the deterministic mode
     // (during testing).
     if (deterministic)
     {
       output = input;
     }
     else
     {
       // Scale with input / (1 - ratio) and set values to zero with probability
       // 'ratio'.
       mask = arma::randu<arma::Mat<eT> >(input.n_rows, input.n_cols);
       mask.transform([&](double val) { return (val > ratio); });
       output = input % mask * scale;
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Dropout<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     g = gy % mask * scale;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void Dropout<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(ratio));
   
     // Reset scale.
     scale = 1.0 / (1.0 - ratio);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
