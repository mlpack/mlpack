
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_margin_ranking_loss_impl.hpp:

Program Listing for File margin_ranking_loss_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_margin_ranking_loss_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/margin_ranking_loss_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MARGIN_IMPL_LOSS_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_MARGIN_IMPL_LOSS_HPP
   
   // In case it hasn't been included.
   #include "margin_ranking_loss.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   MarginRankingLoss<InputDataType, OutputDataType>::MarginRankingLoss(
       const double margin) : margin(margin)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   MarginRankingLoss<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     const int predictionRows = prediction.n_rows;
     const PredictionType& prediction1 = prediction.rows(0,
         predictionRows / 2 - 1);
     const PredictionType& prediction2 = prediction.rows(predictionRows / 2,
         predictionRows - 1);
     return arma::accu(arma::max(arma::zeros(size(target)),
         -target % (prediction1 - prediction2) + margin)) / target.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template <
       typename PredictionType,
       typename TargetType,
       typename LossType
   >
   void MarginRankingLoss<InputDataType, OutputDataType>::Backward(
       const PredictionType& prediction,
       const TargetType& target,
       LossType& loss)
   {
     const int predictionRows = prediction.n_rows;
     const PredictionType& prediction1 = prediction.rows(0,
         predictionRows / 2 - 1);
     const PredictionType& prediction2 = prediction.rows(predictionRows / 2,
         predictionRows - 1);
     loss = -target % (prediction1 - prediction2) + margin;
     loss.elem(arma::find(loss >= 0)).ones();
     loss.elem(arma::find(loss < 0)).zeros();
     loss = (prediction2 - prediction1) % loss / target.n_cols;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void MarginRankingLoss<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(margin));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
