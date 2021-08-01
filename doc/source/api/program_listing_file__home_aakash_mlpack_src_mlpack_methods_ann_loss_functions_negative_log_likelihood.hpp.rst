
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_negative_log_likelihood.hpp:

Program Listing for File negative_log_likelihood.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_negative_log_likelihood.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_HPP
   #define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class NegativeLogLikelihood
   {
    public:
     NegativeLogLikelihood();
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                                const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     InputDataType& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   }; // class NegativeLogLikelihood
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "negative_log_likelihood_impl.hpp"
   
   #endif
