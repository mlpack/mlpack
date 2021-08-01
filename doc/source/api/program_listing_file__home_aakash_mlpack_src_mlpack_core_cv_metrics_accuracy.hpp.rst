
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_metrics_accuracy.hpp:

Program Listing for File accuracy.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_metrics_accuracy.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/metrics/accuracy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_METRICS_ACCURACY_HPP
   #define MLPACK_CORE_CV_METRICS_ACCURACY_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cv {
   
   class Accuracy
   {
    public:
     template<typename MLAlgorithm, typename DataType>
     static double Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const arma::Row<size_t>& labels);
   
     static const bool NeedsMinimization = false;
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation.
   #include "accuracy_impl.hpp"
   
   #endif
