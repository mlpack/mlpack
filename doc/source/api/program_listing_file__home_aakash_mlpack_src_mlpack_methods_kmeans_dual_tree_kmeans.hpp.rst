
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans.hpp:

Program Listing for File dual_tree_kmeans.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/dual_tree_kmeans.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_HPP
   #define MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_HPP
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   #include <mlpack/core/tree/cover_tree.hpp>
   
   #include "dual_tree_kmeans_statistic.hpp"
   
   namespace mlpack {
   namespace kmeans {
   
   template<
       typename MetricType,
       typename MatType,
       template<typename TreeMetricType,
                typename TreeStatType,
                typename TreeMatType>
           class TreeType = tree::KDTree>
   class DualTreeKMeans
   {
    public:
     typedef TreeType<MetricType, DualTreeKMeansStatistic, MatType> Tree;
   
     template<typename TreeMetricType,
              typename IgnoredStatType,
              typename TreeMatType>
     using NNSTreeType =
         TreeType<TreeMetricType, DualTreeKMeansStatistic, TreeMatType>;
   
     DualTreeKMeans(const MatType& dataset, MetricType& metric);
   
     ~DualTreeKMeans();
   
     double Iterate(const arma::mat& centroids,
                    arma::mat& newCentroids,
                    arma::Col<size_t>& counts);
   
     size_t DistanceCalculations() const { return distanceCalculations; }
     size_t& DistanceCalculations() { return distanceCalculations; }
   
    private:
     const MatType& datasetOrig; // Maybe not necessary.
     Tree* tree;
     const MatType& dataset;
     MetricType metric;
   
     size_t distanceCalculations;
     size_t iteration;
   
     arma::vec upperBounds;
     arma::vec lowerBounds;
     std::vector<bool> prunedPoints;
   
     arma::Row<size_t> assignments;
   
     std::vector<bool> visited; // Was the point visited this iteration?
   
     arma::mat lastIterationCentroids; // For sanity checks.
   
     arma::vec clusterDistances; // The amount the clusters moved last iteration.
   
     arma::mat interclusterDistances; // Static storage for intercluster distances.
   
     void UpdateTree(Tree& node,
                     const arma::mat& centroids,
                     const double parentUpperBound = 0.0,
                     const double adjustedParentUpperBound = DBL_MAX,
                     const double parentLowerBound = DBL_MAX,
                     const double adjustedParentLowerBound = 0.0);
   
     void ExtractCentroids(Tree& node,
                           arma::mat& newCentroids,
                           arma::Col<size_t>& newCounts,
                           const arma::mat& centroids);
   
     void CoalesceTree(Tree& node, const size_t child = 0);
     void DecoalesceTree(Tree& node);
   };
   
   template<typename TreeType>
   void HideChild(TreeType& node,
                  const size_t child,
                  const typename std::enable_if_t<
                      !tree::TreeTraits<TreeType>::BinaryTree>* junk = 0);
   
   template<typename TreeType>
   void HideChild(TreeType& node,
                  const size_t child,
                  const typename std::enable_if_t<
                      tree::TreeTraits<TreeType>::BinaryTree>* junk = 0);
   
   template<typename TreeType>
   void RestoreChildren(TreeType& node,
                        const typename std::enable_if_t<!tree::TreeTraits<
                            TreeType>::BinaryTree>* junk = 0);
   
   template<typename TreeType>
   void RestoreChildren(TreeType& node,
                        const typename std::enable_if_t<tree::TreeTraits<
                            TreeType>::BinaryTree>* junk = 0);
   
   template<typename MetricType, typename MatType>
   using DefaultDualTreeKMeans = DualTreeKMeans<MetricType, MatType>;
   
   template<typename MetricType, typename MatType>
   using CoverTreeDualTreeKMeans = DualTreeKMeans<MetricType, MatType,
       tree::StandardCoverTree>;
   
   } // namespace kmeans
   } // namespace mlpack
   
   #include "dual_tree_kmeans_impl.hpp"
   
   #endif
