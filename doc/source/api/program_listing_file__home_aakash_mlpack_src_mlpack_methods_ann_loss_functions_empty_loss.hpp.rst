
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_empty_loss.hpp:

Program Listing for File empty_loss.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_empty_loss.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_EMPTY_LOSS_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTION_EMPTY_LOSS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class EmptyLoss
   {
    public:
     EmptyLoss();
   
     template<typename PredictionType, typename TargetType>
     double Forward(const PredictionType& input, const TargetType& target);
   
     template<typename PredictionType, typename TargetType, typename LossType>
     void Backward(const PredictionType& prediction,
                   const TargetType& target,
                   LossType& loss);
   }; // class EmptyLoss
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "empty_loss_impl.hpp"
   
   #endif
