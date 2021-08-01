
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_naive_kmeans.hpp:

Program Listing for File naive_kmeans.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_naive_kmeans.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/naive_kmeans.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_NAIVE_KMEANS_HPP
   #define MLPACK_METHODS_KMEANS_NAIVE_KMEANS_HPP
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   template<typename MetricType, typename MatType>
   class NaiveKMeans
   {
    public:
     NaiveKMeans(const MatType& dataset, MetricType& metric);
   
     double Iterate(const arma::mat& centroids,
                    arma::mat& newCentroids,
                    arma::Col<size_t>& counts);
   
     size_t DistanceCalculations() const { return distanceCalculations; }
   
    private:
     const MatType& dataset;
     MetricType& metric;
   
     size_t distanceCalculations;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   // Include implementation.
   #include "naive_kmeans_impl.hpp"
   
   #endif
