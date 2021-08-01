
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_metrics_non_maximal_supression.hpp:

Program Listing for File non_maximal_supression.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_metrics_non_maximal_supression.hpp>` (``/home/aakash/mlpack/src/mlpack/core/metrics/non_maximal_supression.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_METRICS_NMS_HPP
   #define MLPACK_CORE_METRICS_NMS_HPP
   
   namespace mlpack {
   namespace metric {
   
   template<bool UseCoordinates = false>
   class NMS
   {
    public:
     NMS() { /* Nothing to do here. */ }
   
     template<
         typename BoundingBoxesType,
         typename ConfidenceScoreType,
         typename OutputType
     >
     static void Evaluate(const BoundingBoxesType& boundingBoxes,
                          const ConfidenceScoreType& confidenceScores,
                          OutputType& selectedIndices,
                          const double threshold = 0.5);
   
     static const bool useCoordinates = UseCoordinates;
   
     template <typename Archive>
     void serialize(Archive &ar, const uint32_t /* version */);
   }; // Class NMS.
   
   } // namespace metric
   } // namespace mlpack
   
   // Include implementation.
   #include "non_maximal_supression_impl.hpp"
   
   #endif
