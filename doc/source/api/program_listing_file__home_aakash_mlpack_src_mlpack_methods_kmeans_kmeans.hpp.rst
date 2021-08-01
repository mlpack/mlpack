
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_kmeans.hpp:

Program Listing for File kmeans.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_kmeans.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/kmeans.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_KMEANS_HPP
   #define MLPACK_METHODS_KMEANS_KMEANS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <mlpack/core/metrics/lmetric.hpp>
   #include "sample_initialization.hpp"
   #include "max_variance_new_cluster.hpp"
   #include "naive_kmeans.hpp"
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   
   namespace mlpack {
   namespace kmeans  {
   
   template<typename MetricType = metric::EuclideanDistance,
            typename InitialPartitionPolicy = SampleInitialization,
            typename EmptyClusterPolicy = MaxVarianceNewCluster,
            template<class, class> class LloydStepType = NaiveKMeans,
            typename MatType = arma::mat>
   class KMeans
   {
    public:
     KMeans(const size_t maxIterations = 1000,
            const MetricType metric = MetricType(),
            const InitialPartitionPolicy partitioner = InitialPartitionPolicy(),
            const EmptyClusterPolicy emptyClusterAction = EmptyClusterPolicy());
   
   
     void Cluster(const MatType& data,
                  const size_t clusters,
                  arma::Row<size_t>& assignments,
                  const bool initialGuess = false);
   
     void Cluster(const MatType& data,
                  size_t clusters,
                  arma::mat& centroids,
                  const bool initialGuess = false);
   
     void Cluster(const MatType& data,
                  const size_t clusters,
                  arma::Row<size_t>& assignments,
                  arma::mat& centroids,
                  const bool initialAssignmentGuess = false,
                  const bool initialCentroidGuess = false);
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     const MetricType& Metric() const { return metric; }
     MetricType& Metric() { return metric; }
   
     const InitialPartitionPolicy& Partitioner() const { return partitioner; }
     InitialPartitionPolicy& Partitioner() { return partitioner; }
   
     const EmptyClusterPolicy& EmptyClusterAction() const
     { return emptyClusterAction; }
     EmptyClusterPolicy& EmptyClusterAction() { return emptyClusterAction; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     size_t maxIterations;
     MetricType metric;
     InitialPartitionPolicy partitioner;
     EmptyClusterPolicy emptyClusterAction;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   // Include implementation.
   #include "kmeans_impl.hpp"
   
   #endif // MLPACK_METHODS_KMEANS_KMEANS_HPP
