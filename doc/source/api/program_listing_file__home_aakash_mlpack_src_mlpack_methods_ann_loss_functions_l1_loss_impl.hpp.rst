
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_l1_loss_impl.hpp:

Program Listing for File l1_loss_impl.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_l1_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/l1_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "l1_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   L1Loss<InputDataType, OutputDataType>::L1Loss(const bool mean):
     mean(mean)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   L1Loss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     if (mean)
       return arma::accu(arma::mean(prediction - target));
   
     return arma::accu(prediction - target);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void L1Loss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     loss = arma::sign(prediction - target);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void L1Loss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(mean));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
