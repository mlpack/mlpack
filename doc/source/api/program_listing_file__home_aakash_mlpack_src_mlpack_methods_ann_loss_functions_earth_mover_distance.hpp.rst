
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_earth_mover_distance.hpp:

Program Listing for File earth_mover_distance.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_loss_functions_earth_mover_distance.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/loss_functions/earth_mover_distance.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_HPP
   #define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_EARTH_MOVER_DISTANCE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class EarthMoverDistance
   {
    public:
     EarthMoverDistance();
   
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
   }; // class EarthMoverDistance
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "earth_mover_distance_impl.hpp"
   
   #endif
