
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_mean_bias_error_impl.hpp:

Program Listing for File mean_bias_error_impl.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_mean_bias_error_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/mean_bias_error_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_BIAS_ERROR_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_BIAS_ERROR_IMPL_HPP
   
   
   // In case it hasn't yet been included.
   #include "mean_bias_error.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   MeanBiasError<InputDataType, OutputDataType>::MeanBiasError()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   MeanBiasError<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     return arma::accu(target - prediction) / target.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void MeanBiasError<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& /* target */,
       LossType& loss)
   {
     loss.set_size(arma::size(prediction));
     loss.fill(-1.0);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void MeanBiasError<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
