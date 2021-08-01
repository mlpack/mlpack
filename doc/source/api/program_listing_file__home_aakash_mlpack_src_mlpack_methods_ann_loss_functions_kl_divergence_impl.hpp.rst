
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_kl_divergence_impl.hpp:

Program Listing for File kl_divergence_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_kl_divergence_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/kl_divergence_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "kl_divergence.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   KLDivergence<InputDataType, OutputDataType>::KLDivergence(const bool takeMean) :
       takeMean(takeMean)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   KLDivergence<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     if (takeMean)
     {
       return arma::as_scalar(arma::mean(
           arma::mean(prediction % (arma::log(prediction) - arma::log(target)))));
     }
     else
     {
       return arma::accu(prediction % (arma::log(prediction) - arma::log(target)));
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void KLDivergence<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     if (takeMean)
     {
       loss = arma::mean(arma::mean(
           arma::log(prediction) - arma::log(target) + 1));
     }
     else
     {
       loss = arma::accu(arma::log(prediction) - arma::log(target) + 1);
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void KLDivergence<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(takeMean));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
