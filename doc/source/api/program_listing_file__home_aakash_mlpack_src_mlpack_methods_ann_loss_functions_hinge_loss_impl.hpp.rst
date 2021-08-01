
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_hinge_loss_impl.hpp:

Program Listing for File hinge_loss_impl.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_hinge_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/hinge_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_LOSS_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_LOSS_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "hinge_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   HingeLoss<InputDataType, OutputDataType>::HingeLoss(const bool reduction):
     reduction(reduction)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   HingeLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     TargetType temp = target - (target == 0);
     TargetType temp_zeros(size(target), arma::fill::zeros);
   
     PredictionType loss = arma::max(temp_zeros, 1 - prediction % temp);
   
     typename PredictionType::elem_type lossSum = arma::accu(loss);
   
     if (reduction)
       return lossSum;
   
     return lossSum / loss.n_elem;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void HingeLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     TargetType temp = target - (target == 0);
     loss = (prediction < (1 / temp)) % -temp;
   
     if (!reduction)
       loss /= target.n_elem;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void HingeLoss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(reduction));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
