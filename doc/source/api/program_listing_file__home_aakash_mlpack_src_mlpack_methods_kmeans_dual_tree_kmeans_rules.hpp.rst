
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans_rules.hpp:

Program Listing for File dual_tree_kmeans_rules.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_dual_tree_kmeans_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/dual_tree_kmeans_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_HPP
   #define MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_HPP
   
   #include <mlpack/core/tree/traversal_info.hpp>
   
   namespace mlpack {
   namespace kmeans {
   
   template<typename MetricType, typename TreeType>
   class DualTreeKMeansRules
   {
    public:
     DualTreeKMeansRules(const arma::mat& centroids,
                         const arma::mat& dataset,
                         arma::Row<size_t>& assignments,
                         arma::vec& upperBounds,
                         arma::vec& lowerBounds,
                         MetricType& metric,
                         const std::vector<bool>& prunedPoints,
                         const std::vector<size_t>& oldFromNewCentroids,
                         std::vector<bool>& visited);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
     double Score(TreeType& queryNode, TreeType& referenceNode);
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore);
     double Rescore(TreeType& queryNode,
                    TreeType& referenceNode,
                    const double oldScore);
   
     typedef typename tree::TraversalInfo<TreeType> TraversalInfoType;
   
     TraversalInfoType& TraversalInfo() { return traversalInfo; }
     const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
   
     size_t BaseCases() const { return baseCases; }
     size_t& BaseCases() { return baseCases; }
   
     size_t Scores() const { return scores; }
     size_t& Scores() { return scores; }
   
     size_t MinimumBaseCases() const { return 0; }
   
    private:
     const arma::mat& centroids;
     const arma::mat& dataset;
     arma::Row<size_t>& assignments;
     arma::vec& upperBounds;
     arma::vec& lowerBounds;
     MetricType& metric;
   
     const std::vector<bool>& prunedPoints;
   
     const std::vector<size_t>& oldFromNewCentroids;
   
     std::vector<bool>& visited;
   
     size_t baseCases;
     size_t scores;
   
     TraversalInfoType traversalInfo;
   
     size_t lastQueryIndex;
     size_t lastReferenceIndex;
     size_t lastBaseCase;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   #include "dual_tree_kmeans_rules_impl.hpp"
   
   #endif
