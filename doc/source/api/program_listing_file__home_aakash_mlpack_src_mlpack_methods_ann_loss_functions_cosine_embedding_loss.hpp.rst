
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_cosine_embedding_loss.hpp:

Program Listing for File cosine_embedding_loss.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_cosine_embedding_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/cosine_embedding_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_COSINE_EMBEDDING_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class CosineEmbeddingLoss
   {
    public:
     CosineEmbeddingLoss(const double margin = 0.0,
                         const bool similarity = true,
                         const bool takeMean = false);
   
     template <typename PredictionType, typename TargetType>
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
   
     bool TakeMean() const { return takeMean; }
     bool& TakeMean() { return takeMean; }
   
     double Margin() const { return margin; }
     double& Margin() { return margin; }
   
     bool Similarity() const { return similarity; }
     bool& Similarity() { return similarity; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     InputDataType inputParameter;
   
     OutputDataType outputParameter;
   
     double margin;
   
     bool similarity;
   
     bool takeMean;
   }; // class CosineEmbeddingLoss
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "cosine_embedding_loss_impl.hpp"
   
   #endif
