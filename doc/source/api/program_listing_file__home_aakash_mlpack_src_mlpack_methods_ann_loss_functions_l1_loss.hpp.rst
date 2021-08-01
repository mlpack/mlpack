
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_l1_loss.hpp:

Program Listing for File l1_loss.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_l1_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/l1_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class L1Loss
   {
    public:
     L1Loss(const bool mean = true);
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                                const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     bool Mean() const { return mean; }
     bool& Mean() { return mean; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType outputParameter;
   
     bool mean;
   }; // class L1Loss
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "l1_loss_impl.hpp"
   
   #endif
