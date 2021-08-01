
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_metrics_facilities.hpp:

Program Listing for File facilities.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_metrics_facilities.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/metrics/facilities.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_METRICS_FACILITIES_HPP
   #define MLPACK_CORE_CV_METRICS_FACILITIES_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   
   namespace mlpack {
   namespace cv {
   
   template<typename DataType, typename Metric>
   DataType PairwiseDistances(const DataType& data,
                              const Metric& metric)
   {
     DataType distances = DataType(data.n_cols, data.n_cols, arma::fill::none);
     for (size_t i = 0; i < data.n_cols; i++)
     {
       for (size_t j = 0; j < i; j++)
       {
         distances(i, j) = metric.Evaluate(data.col(i), data.col(j));
         distances(j, i) = distances(i, j);
       }
     }
     distances.diag().zeros();
     return distances;
   }
   
   } // namespace cv
   } // namespace mlpack
   
   #endif
