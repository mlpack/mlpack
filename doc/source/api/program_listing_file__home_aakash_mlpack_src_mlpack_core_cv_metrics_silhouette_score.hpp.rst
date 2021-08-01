
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_metrics_silhouette_score.hpp:

Program Listing for File silhouette_score.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_metrics_silhouette_score.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/metrics/silhouette_score.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_METRICS_SILHOUETTE_SCORE_HPP
   #define MLPACK_CORE_CV_METRICS_SILHOUETTE_SCORE_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace cv {
   
   class SilhouetteScore
   {
    public:
     template<typename DataType, typename Metric>
     static double Overall(const DataType& X,
                           const arma::Row<size_t>& labels,
                           const Metric& metric);
   
     template<typename DataType>
     static arma::rowvec SamplesScore(const DataType& distances,
                                      const arma::Row<size_t>& labels);
   
     template<typename DataType, typename Metric>
     static arma::rowvec SamplesScore(const DataType& X,
                                      const arma::Row<size_t>& labels,
                                      const Metric& metric);
   
     static double MeanDistanceFromCluster(const arma::colvec& distances,
                                           const arma::Row<size_t>& labels,
                                           const size_t& label,
                                           const bool& sameCluster = false);
   
     static const bool NeedsMinimization = false;
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation.
   #include "silhouette_score_impl.hpp"
   
   #endif
