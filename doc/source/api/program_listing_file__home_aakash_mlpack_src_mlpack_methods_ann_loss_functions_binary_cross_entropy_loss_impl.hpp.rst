
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_binary_cross_entropy_loss_impl.hpp:

Program Listing for File binary_cross_entropy_loss_impl.hpp
===========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_binary_cross_entropy_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/binary_cross_entropy_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_ERROR_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_ERROR_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "binary_cross_entropy_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   BCELoss<InputDataType, OutputDataType>::BCELoss(
       const double eps, const bool reduction) : eps(eps), reduction(reduction)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   BCELoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     typedef typename PredictionType::elem_type ElemType;
   
     ElemType loss = -arma::accu(target % arma::log(prediction + eps) +
         (1. - target) % arma::log(1. - prediction + eps));
     if (reduction)
       loss /= prediction.n_elem;
     return loss;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void BCELoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     loss = (1. - target) / (1. - prediction + eps) - target / (prediction + eps);
     if (reduction)
       loss /= prediction.n_elem;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void BCELoss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(eps));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
