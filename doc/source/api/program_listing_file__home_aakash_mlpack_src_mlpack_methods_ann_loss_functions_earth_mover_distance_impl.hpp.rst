
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_earth_mover_distance_impl.hpp:

Program Listing for File earth_mover_distance_impl.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_earth_mover_distance_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/earth_mover_distance_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_IMPL_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "earth_mover_distance.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   EarthMoverDistance<InputDataType, OutputDataType>::EarthMoverDistance()
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType>
   typename PredictionType::elem_type
   EarthMoverDistance<InputDataType, OutputDataType>::Forward(
       const PredictionType& prediction,
       const TargetType& target)
   {
     return -arma::accu(target % prediction);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename PredictionType, typename TargetType, typename LossType>
   void EarthMoverDistance<InputDataType, OutputDataType>::Backward(
       const PredictionType& /* prediction */,
       const TargetType& target,
       LossType& loss)
   {
     loss = -target;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void EarthMoverDistance<InputDataType, OutputDataType>::serialize(
       Archive& /* ar */,
       const uint32_t /* version */)
   {
     /* Nothing to do here */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
