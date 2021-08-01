
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_binary_cross_entropy_loss.hpp:

Program Listing for File binary_cross_entropy_loss.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_binary_cross_entropy_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_ERROR_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_CROSS_ENTROPY_ERROR_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class BCELoss
   {
    public:
     BCELoss(const double eps = 1e-10, const bool reduction = true);
   
     template<typename PredictionType, typename TargetType>
     typename PredictionType::elem_type Forward(const PredictionType& prediction,
                                                const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     double Eps() const { return eps; }
     double& Eps() { return eps; }
   
     bool Reduction() const { return reduction; }
     bool& Reduction() { return reduction; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType outputParameter;
   
     double eps;
   
     bool reduction;
   }; // class BCELoss
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using CrossEntropyError = BCELoss<
       InputDataType, OutputDataType>;
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "binary_cross_entropy_loss_impl.hpp"
   
   #endif
