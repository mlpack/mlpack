
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_triplet_margin_loss_impl.hpp:

Program Listing for File triplet_margin_loss_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_triplet_margin_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/triplet_margin_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_IMPL_LOSS_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_IMPL_LOSS_HPP
   
   // In case it hasn't been included.
   #include "triplet_margin_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   TripletMarginLoss<InputDataType, OutputDataType>::TripletMarginLoss(
       const double margin) : margin(margin)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   TripletMarginLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     PredictionType anchor =
         prediction.submat(0, 0, prediction.n_rows / 2 - 1, prediction.n_cols - 1);
     PredictionType positive =
         prediction.submat(prediction.n_rows / 2, 0, prediction.n_rows - 1,
         prediction.n_cols - 1);
     return std::max(0.0, arma::accu(arma::pow(anchor - positive, 2)) -
         arma::accu(arma::pow(anchor - target, 2)) + margin) / anchor.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template <
       typename PredictionType,
       typename TargetType,
       typename LossType
   >
   void TripletMarginLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     PredictionType positive =
         prediction.submat(prediction.n_rows / 2, 0, prediction.n_rows - 1,
         prediction.n_cols - 1);
     loss = 2 * (target - positive) / target.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void TripletMarginLoss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const unsigned int /* version */)
   {
     ar(CEREAL_NVP(margin));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
