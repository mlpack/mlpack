
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_mean_squared_error_impl.hpp:

Program Listing for File mean_squared_error_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_mean_squared_error_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "mean_squared_error.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   MeanSquaredError<InputDataType, OutputDataType>::MeanSquaredError()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   MeanSquaredError<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     return arma::accu(arma::square(prediction - target)) / target.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void MeanSquaredError<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     loss = 2 * (prediction - target) / target.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void MeanSquaredError<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
