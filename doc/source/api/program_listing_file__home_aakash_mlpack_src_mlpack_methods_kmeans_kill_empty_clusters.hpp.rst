
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_kill_empty_clusters.hpp:

Program Listing for File kill_empty_clusters.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_kill_empty_clusters.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/kill_empty_clusters.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef __MLPACK_METHODS_KMEANS_KILL_EMPTY_CLUSTERS_HPP
   #define __MLPACK_METHODS_KMEANS_KILL_EMPTY_CLUSTERS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   class KillEmptyClusters
   {
    public:
     KillEmptyClusters() { }
   
     template<typename MetricType, typename MatType>
     static inline force_inline void EmptyCluster(
         const MatType& /* data */,
         const size_t emptyCluster,
         const arma::mat& /* oldCentroids */,
         arma::mat& newCentroids,
         arma::Col<size_t>& clusterCounts,
         MetricType& /* metric */,
         const size_t /* iteration */)
   {
     // Remove the empty cluster.
     if (emptyCluster < newCentroids.n_cols)
     {
       newCentroids.shed_col(emptyCluster);
       clusterCounts.shed_row(emptyCluster);
     }
   }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   #endif
