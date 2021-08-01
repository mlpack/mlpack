
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_mean_absolute_percentage_error_impl.hpp:

Program Listing for File mean_absolute_percentage_error_impl.hpp
================================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_mean_absolute_percentage_error_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/mean_absolute_percentage_error_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "mean_absolute_percentage_error.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   MeanAbsolutePercentageError<InputDataType, OutputDataType>::
   MeanAbsolutePercentageError()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   MeanAbsolutePercentageError<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     PredictionType loss = arma::abs((prediction - target) / target);
     return arma::accu(loss) * (100 / target.n_cols);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void MeanAbsolutePercentageError<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   
   {
     loss = (((arma::conv_to<arma::mat>::from(prediction < target) * -2) + 1) /
         target) * (100 / target.n_cols);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void MeanAbsolutePercentageError<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const unsigned int /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
