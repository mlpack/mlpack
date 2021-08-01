
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_triplet_margin_loss.hpp:

Program Listing for File triplet_margin_loss.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_triplet_margin_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/triplet_margin_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_LOSS_HPP
   #define MLPACK_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_LOSS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class TripletMarginLoss
   {
    public:
     TripletMarginLoss(const double margin = 1.0);
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                           const TargetType& target);
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     double Margin() const { return margin; }
     double& Margin() { return margin; }
   
     template<typename Archive>
     void serialize(Archive& ar, const unsigned int /* version */);
   
    private:
     OutputDataType outputParameter;
   
     double margin;
   }; // class TripletLossMargin
   
   } // namespace ann
   } // namespace mlpack
   
   // include implementation.
   #include "triplet_margin_loss_impl.hpp"
   
   #endif
