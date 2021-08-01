
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_hinge_embedding_loss.hpp:

Program Listing for File hinge_embedding_loss.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_hinge_embedding_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/hinge_embedding_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_EMBEDDING_LOSS_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_EMBEDDING_LOSS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
           typename InputDataType = arma::mat,
           typename OutputDataType = arma::mat
   >
   class HingeEmbeddingLoss
   {
    public:
     HingeEmbeddingLoss();
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                                const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType outputParameter;
   }; // class HingeEmbeddingLoss
   
   } // namespace ann
   } // namespace mlpack
   
   // include implementation
   #include "hinge_embedding_loss_impl.hpp"
   
   #endif
