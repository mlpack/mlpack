
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_pelleg_moore_kmeans.hpp:

Program Listing for File pelleg_moore_kmeans.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_pelleg_moore_kmeans.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/pelleg_moore_kmeans.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_HPP
   #define MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_HPP
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include "pelleg_moore_kmeans_statistic.hpp"
   
   namespace mlpack {
   namespace kmeans {
   
   template<typename MetricType, typename MatType>
   class PellegMooreKMeans
   {
    public:
     PellegMooreKMeans(const MatType& dataset, MetricType& metric);
   
     ~PellegMooreKMeans();
   
     double Iterate(const arma::mat& centroids,
                    arma::mat& newCentroids,
                    arma::Col<size_t>& counts);
   
     size_t DistanceCalculations() const { return distanceCalculations; }
     size_t& DistanceCalculations() { return distanceCalculations; }
   
     typedef tree::KDTree<MetricType, PellegMooreKMeansStatistic, MatType>
         TreeType;
   
    private:
     const MatType& datasetOrig; // Maybe not necessary.
     TreeType* tree;
     const MatType& dataset;
     MetricType& metric;
   
     size_t distanceCalculations;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   #include "pelleg_moore_kmeans_impl.hpp"
   
   #endif
