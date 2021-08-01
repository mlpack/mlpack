
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_empty_loss_impl.hpp:

Program Listing for File empty_loss_impl.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_empty_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/empty_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EMPTY_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EMPTY_LOSS_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "empty_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   EmptyLoss<InputDataType, OutputDataType>::EmptyLoss()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   double EmptyLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& /* prediction */, const TargetType& /* target */)
   {
     return 0;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void EmptyLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& /* prediction */,
       const TargetType& target,
       LossType& loss)
   {
     loss = target;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
