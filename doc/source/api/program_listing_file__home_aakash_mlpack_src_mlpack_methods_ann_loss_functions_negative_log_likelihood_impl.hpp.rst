
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_negative_log_likelihood_impl.hpp:

Program Listing for File negative_log_likelihood_impl.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_negative_log_likelihood_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/negative_log_likelihood_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "negative_log_likelihood.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   NegativeLogLikelihood<InputDataType, OutputDataType>::NegativeLogLikelihood()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   NegativeLogLikelihood<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     typedef typename PredictionType::elem_type ElemType;
     ElemType output = 0;
     for (size_t i = 0; i < prediction.n_cols; ++i)
     {
       Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
           "Target class out of range.");
   
       output -= prediction(target(i), i);
     }
   
     return output;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void NegativeLogLikelihood<InputDataType, OutputDataType>::Backward(
         const PredictionType& prediction,
         const TargetType& target,
         LossType& loss)
   {
     loss = arma::zeros<LossType>(prediction.n_rows, prediction.n_cols);
     for (size_t i = 0; i < prediction.n_cols; ++i)
     {
       Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
           "Target class out of range.");
   
       loss(target(i), i) = -1;
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void NegativeLogLikelihood<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */, const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
