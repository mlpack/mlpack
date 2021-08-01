
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_huber_loss_impl.hpp:

Program Listing for File huber_loss_impl.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_huber_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/huber_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "huber_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   HuberLoss<InputDataType, OutputDataType>::HuberLoss(
     const double delta,
     const bool mean):
     delta(delta),
     mean(mean)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   HuberLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     typedef typename PredictionType::elem_type ElemType;
     ElemType loss = 0;
     for (size_t i = 0; i < prediction.n_elem; ++i)
     {
         const ElemType absError = std::abs(target[i] - prediction[i]);
         loss += absError > delta ?
             delta * (absError - 0.5 * delta) : 0.5 * std::pow(absError, 2);
     }
     return mean ? loss / prediction.n_elem : loss;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void HuberLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     typedef typename PredictionType::elem_type ElemType;
   
     loss.set_size(size(prediction));
     for (size_t i = 0; i < loss.n_elem; ++i)
     {
       const ElemType absError = std::abs(target[i] - prediction[i]);
       loss[i] = absError > delta ?
           -delta * (target[i] - prediction[i]) / absError :
           prediction[i] - target[i];
       if (mean)
         loss[i] /= loss.n_elem;
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void HuberLoss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(delta));
     ar(CEREAL_NVP(mean));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
