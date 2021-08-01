
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_elkan_kmeans.hpp:

Program Listing for File elkan_kmeans.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_elkan_kmeans.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/elkan_kmeans.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_ELKAN_KMEANS_HPP
   #define MLPACK_METHODS_KMEANS_ELKAN_KMEANS_HPP
   
   namespace mlpack {
   namespace kmeans {
   
   template<typename MetricType, typename MatType>
   class ElkanKMeans
   {
    public:
     ElkanKMeans(const MatType& dataset, MetricType& metric);
   
     double Iterate(const arma::mat& centroids,
                    arma::mat& newCentroids,
                    arma::Col<size_t>& counts);
   
     size_t DistanceCalculations() const { return distanceCalculations; }
   
    private:
     const MatType& dataset;
     MetricType& metric;
   
     arma::mat clusterDistances;
     arma::vec minClusterDistances;
   
     arma::Col<size_t> assignments;
   
     arma::vec upperBounds;
     arma::mat lowerBounds;
   
     size_t distanceCalculations;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   // Include implementation.
   #include "elkan_kmeans_impl.hpp"
   
   #endif
