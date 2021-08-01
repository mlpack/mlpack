
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_reconstruction_loss_impl.hpp:

Program Listing for File reconstruction_loss_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_reconstruction_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/reconstruction_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_RECONSTRUCTION_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_RECONSTRUCTION_LOSS_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "reconstruction_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType, typename DistType>
   ReconstructionLoss<
       InputDataType,
       OutputDataType,
       DistType
   >::ReconstructionLoss()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType, typename DistType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   ReconstructionLoss<InputDataType, OutputDataType, DistType>::Forward(
       const PredictionType& prediction, const TargetType& target)
   {
     dist = DistType(prediction);
     return -dist.LogProbability(target);
   }
   
   template<typename InputDataType, typename OutputDataType, typename DistType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void ReconstructionLoss<InputDataType, OutputDataType, DistType>::Backward(
       const PredictionType& /* prediction */,
       const TargetType& target,
       LossType& loss)
   {
     dist.LogProbBackward(target, loss);
     loss *= -1;
   }
   
   template<typename InputDataType, typename OutputDataType, typename DistType>
   template<typename Archive>
   void ReconstructionLoss<InputDataType, OutputDataType, DistType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     // Nothing to do here.
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
