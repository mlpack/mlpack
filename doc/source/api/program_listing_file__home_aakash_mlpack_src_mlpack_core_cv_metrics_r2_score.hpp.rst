
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_metrics_r2_score.hpp:

Program Listing for File r2_score.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_metrics_r2_score.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/metrics/r2_score.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_METRICS_R2SCORE_HPP
   #define MLPACK_CORE_CV_METRICS_R2SCORE_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace cv {
   
   template<bool AdjustedR2>
   class R2Score
   {
    public:
     template<typename MLAlgorithm, typename DataType, typename ResponsesType>
     static double Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const ResponsesType& responses);
   
     static const bool NeedsMinimization = false;
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation.
   #include "r2_score_impl.hpp"
   
   #endif
