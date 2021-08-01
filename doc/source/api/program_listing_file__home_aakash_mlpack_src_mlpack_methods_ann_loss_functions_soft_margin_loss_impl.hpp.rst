
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_soft_margin_loss_impl.hpp:

Program Listing for File soft_margin_loss_impl.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_soft_margin_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/soft_margin_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_IMPL_HPP
   
   // In case it hasn't been included.
   #include "soft_margin_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   SoftMarginLoss<InputDataType, OutputDataType>::
   SoftMarginLoss(const bool reduction) : reduction(reduction)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   SoftMarginLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction, const TargetType& target)
   {
     PredictionType loss = arma::log(1 + arma::exp(-target % prediction));
     typename PredictionType::elem_type lossSum = arma::accu(loss);
   
     if (reduction)
       return lossSum;
   
     return lossSum / prediction.n_elem;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void SoftMarginLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     loss.set_size(size(prediction));
     PredictionType temp = arma::exp(-target % prediction);
     PredictionType numerator = -target % temp;
     PredictionType denominator = 1 + temp;
     loss = numerator / denominator;
   
     if (!reduction)
       loss = loss / prediction.n_elem;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void SoftMarginLoss<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(reduction));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
