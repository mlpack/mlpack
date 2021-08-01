
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_kl_divergence.hpp:

Program Listing for File kl_divergence.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_kl_divergence.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/kl_divergence.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_KL_DIVERGENCE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
           typename InputDataType = arma::mat,
           typename OutputDataType = arma::mat
   >
   class KLDivergence
   {
    public:
     KLDivergence(const bool takeMean = false);
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                                const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     bool TakeMean() const { return takeMean; }
     bool& TakeMean() { return takeMean; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType outputParameter;
   
     bool takeMean;
   }; // class KLDivergence
   
   } // namespace ann
   } // namespace mlpack
   
   // include implementation
   #include "kl_divergence_impl.hpp"
   
   #endif
