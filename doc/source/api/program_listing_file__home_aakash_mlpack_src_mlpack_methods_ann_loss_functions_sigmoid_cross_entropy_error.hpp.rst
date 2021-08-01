
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_sigmoid_cross_entropy_error.hpp:

Program Listing for File sigmoid_cross_entropy_error.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_sigmoid_cross_entropy_error.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_SIGMOID_CROSS_ENTROPY_ERROR_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class SigmoidCrossEntropyError
   {
    public:
     SigmoidCrossEntropyError();
   
     template<typename PredictionType, typename TargetType>
     inline typename PredictionType::elem_type Forward(
         const PredictionType& prediction,
         const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     inline void Backward(const PredictionType& prediction,
                          const TargetType& target,
                          LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType outputParameter;
   }; // class SigmoidCrossEntropy
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "sigmoid_cross_entropy_error_impl.hpp"
   
   #endif
