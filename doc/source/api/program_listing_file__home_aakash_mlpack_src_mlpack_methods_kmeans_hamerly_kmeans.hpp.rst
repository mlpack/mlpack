
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_hamerly_kmeans.hpp:

Program Listing for File hamerly_kmeans.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_hamerly_kmeans.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/hamerly_kmeans.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_HPP
   #define MLPACK_METHODS_KMEANS_HAMERLY_KMEANS_HPP
   
   namespace mlpack {
   namespace kmeans {
   
   template<typename MetricType, typename MatType>
   class HamerlyKMeans
   {
    public:
     HamerlyKMeans(const MatType& dataset, MetricType& metric);
   
     double Iterate(const arma::mat& centroids,
                    arma::mat& newCentroids,
                    arma::Col<size_t>& counts);
   
     size_t DistanceCalculations() const { return distanceCalculations; }
   
    private:
     const MatType& dataset;
     MetricType& metric;
   
     arma::vec minClusterDistances;
   
     arma::vec upperBounds;
     arma::vec lowerBounds;
     arma::Col<size_t> assignments;
   
     size_t distanceCalculations;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   // Include implementation.
   #include "hamerly_kmeans_impl.hpp"
   
   #endif
