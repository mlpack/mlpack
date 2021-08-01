
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_dice_loss.hpp:

Program Listing for File dice_loss.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_dice_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/dice_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class DiceLoss
   {
    public:
     DiceLoss(const double smooth = 1);
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                                const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     double Smooth() const { return smooth; }
     double& Smooth() { return smooth; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType outputParameter;
   
     double smooth;
   }; // class DiceLoss
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "dice_loss_impl.hpp"
   
   #endif
