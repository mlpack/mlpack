
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_allow_empty_clusters.hpp:

Program Listing for File allow_empty_clusters.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_allow_empty_clusters.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/allow_empty_clusters.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_ALLOW_EMPTY_CLUSTERS_HPP
   #define MLPACK_METHODS_KMEANS_ALLOW_EMPTY_CLUSTERS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   class AllowEmptyClusters
   {
    public:
     AllowEmptyClusters() { }
   
     template<typename MetricType, typename MatType>
     static inline force_inline void EmptyCluster(
         const MatType& /* data */,
         const size_t emptyCluster,
         const arma::mat& oldCentroids,
         arma::mat& newCentroids,
         arma::Col<size_t>& /* clusterCounts */,
         MetricType& /* metric */,
         const size_t /* iteration */)
     {
       // Take the last iteration's centroid.
       newCentroids.col(emptyCluster) = oldCentroids.col(emptyCluster);
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   #endif
