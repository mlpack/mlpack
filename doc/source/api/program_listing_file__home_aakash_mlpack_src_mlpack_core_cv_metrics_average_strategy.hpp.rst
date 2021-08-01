
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_metrics_average_strategy.hpp:

Program Listing for File average_strategy.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_metrics_average_strategy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/metrics/average_strategy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_METRICS_AVERAGE_STRATEGY_HPP
   #define MLPACK_CORE_CV_METRICS_AVERAGE_STRATEGY_HPP
   
   namespace mlpack {
   namespace cv {
   
   enum AverageStrategy
   {
     Binary,
     Micro,
     Macro
   };
   
   } // namespace cv
   } // namespace mlpack
   
   #endif
