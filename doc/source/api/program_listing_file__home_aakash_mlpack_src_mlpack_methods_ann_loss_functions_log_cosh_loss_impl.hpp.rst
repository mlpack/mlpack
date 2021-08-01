
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_log_cosh_loss_impl.hpp:

Program Listing for File log_cosh_loss_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_log_cosh_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/log_cosh_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_LOG_COSH_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_LOG_COSH_LOSS_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "log_cosh_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   LogCoshLoss<InputDataType, OutputDataType>::LogCoshLoss(const double a) :
       a(a)
   {
     Log::Assert(a > 0, "Hyper-Parameter \'a\' must be positive");
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   LogCoshLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     return arma::accu(arma::log(arma::cosh(a * (target - prediction)))) / a;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void LogCoshLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     loss = arma::tanh(a * (target - prediction));
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void LogCoshLoss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(a));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
