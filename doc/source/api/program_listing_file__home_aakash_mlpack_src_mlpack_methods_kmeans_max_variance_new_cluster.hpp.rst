
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_max_variance_new_cluster.hpp:

Program Listing for File max_variance_new_cluster.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_max_variance_new_cluster.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/max_variance_new_cluster.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_HPP
   #define MLPACK_METHODS_KMEANS_MAX_VARIANCE_NEW_CLUSTER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   class MaxVarianceNewCluster
   {
    public:
     MaxVarianceNewCluster() : iteration(size_t(-1)) { }
   
     template<typename MetricType, typename MatType>
     void EmptyCluster(const MatType& data,
                       const size_t emptyCluster,
                       const arma::mat& oldCentroids,
                       arma::mat& newCentroids,
                       arma::Col<size_t>& clusterCounts,
                       MetricType& metric,
                       const size_t iteration);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     size_t iteration;
     arma::vec variances;
     arma::Row<size_t> assignments;
   
     template<typename MetricType, typename MatType>
     void Precalculate(const MatType& data,
                       const arma::mat& oldCentroids,
                       arma::Col<size_t>& clusterCounts,
                       MetricType& metric);
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   // Include implementation.
   #include "max_variance_new_cluster_impl.hpp"
   
   #endif
