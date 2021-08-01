
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_metrics_mse.hpp:

Program Listing for File mse.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_metrics_mse.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/metrics/mse.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_METRICS_MSE_HPP
   #define MLPACK_CORE_CV_METRICS_MSE_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace cv {
   
   class MSE
   {
    public:
     template<typename MLAlgorithm, typename DataType, typename ResponsesType>
     static double Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const ResponsesType& responses);
   
     static const bool NeedsMinimization = true;
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation.
   #include "mse_impl.hpp"
   
   #endif
