
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_sigmoid_cross_entropy_error_impl.hpp:

Program Listing for File sigmoid_cross_entropy_error_impl.hpp
=============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_sigmoid_cross_entropy_error_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "sigmoid_cross_entropy_error.hpp"
   #include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   SigmoidCrossEntropyError<InputDataType, OutputDataType>
   ::SigmoidCrossEntropyError()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   inline typename PredictionType::elem_type
   SigmoidCrossEntropyError<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     typedef typename PredictionType::elem_type ElemType;
     ElemType maximum = 0;
     for (size_t i = 0; i < prediction.n_elem; ++i)
     {
       maximum += std::max(prediction[i], 0.0) +
           std::log(1 + std::exp(-std::abs(prediction[i])));
     }
   
     return maximum - arma::accu(prediction % target);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   inline void SigmoidCrossEntropyError<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     loss = 1.0 / (1.0 + arma::exp(-prediction)) - target;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void SigmoidCrossEntropyError<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
