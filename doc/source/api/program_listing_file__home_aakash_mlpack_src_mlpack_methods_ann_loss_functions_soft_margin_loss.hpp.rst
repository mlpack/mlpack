
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_soft_margin_loss.hpp:

Program Listing for File soft_margin_loss.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_soft_margin_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/soft_margin_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_HPP
   #define MLPACK_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class SoftMarginLoss
   {
    public:
     SoftMarginLoss(const bool reduction = true);
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                                const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     bool Reduction() const { return reduction; }
     bool& Reduction() { return reduction; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     OutputDataType outputParameter;
   
     bool reduction;
   }; // class SoftMarginLoss
   
   } // namespace ann
   } // namespace mlpack
   
   // include implementation.
   #include "soft_margin_loss_impl.hpp"
   
   #endif
