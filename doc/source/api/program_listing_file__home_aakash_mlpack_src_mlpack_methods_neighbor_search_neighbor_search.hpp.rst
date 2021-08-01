
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search.hpp:

Program Listing for File neighbor_search.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/neighbor_search.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <vector>
   #include <string>
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   #include <mlpack/core/tree/rectangle_tree.hpp>
   #include <mlpack/core/tree/binary_space_tree/binary_space_tree.hpp>
   
   #include "neighbor_search_stat.hpp"
   #include "sort_policies/nearest_neighbor_sort.hpp"
   #include "neighbor_search_rules.hpp"
   
   namespace mlpack {
   // Neighbor-search routines. These include all-nearest-neighbors and
   // all-furthest-neighbors searches.
   namespace neighbor  {
   
   // Forward declaration.
   template<typename SortPolicy,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType,
            template<typename RuleType> class DualTreeTraversalType,
            template<typename RuleType> class SingleTreeTraversalType>
   class LeafSizeNSWrapper;
   
   enum NeighborSearchMode
   {
     NAIVE_MODE,
     SINGLE_TREE_MODE,
     DUAL_TREE_MODE,
     GREEDY_SINGLE_TREE_MODE
   };
   
   template<typename SortPolicy = NearestNeighborSort,
            typename MetricType = mlpack::metric::EuclideanDistance,
            typename MatType = arma::mat,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType = tree::KDTree,
            template<typename RuleType> class DualTreeTraversalType =
                TreeType<MetricType,
                         NeighborSearchStat<SortPolicy>,
                         MatType>::template DualTreeTraverser,
            template<typename RuleType> class SingleTreeTraversalType =
                TreeType<MetricType,
                         NeighborSearchStat<SortPolicy>,
                         MatType>::template SingleTreeTraverser>
   class NeighborSearch
   {
    public:
     typedef TreeType<MetricType, NeighborSearchStat<SortPolicy>, MatType> Tree;
   
     NeighborSearch(MatType referenceSet,
                    const NeighborSearchMode mode = DUAL_TREE_MODE,
                    const double epsilon = 0,
                    const MetricType metric = MetricType());
   
     NeighborSearch(Tree referenceTree,
                    const NeighborSearchMode mode = DUAL_TREE_MODE,
                    const double epsilon = 0,
                    const MetricType metric = MetricType());
   
     NeighborSearch(const NeighborSearchMode mode = DUAL_TREE_MODE,
                    const double epsilon = 0,
                    const MetricType metric = MetricType());
   
     NeighborSearch(const NeighborSearch& other);
   
     NeighborSearch(NeighborSearch&& other);
   
     NeighborSearch& operator=(const NeighborSearch& other);
   
     NeighborSearch& operator=(NeighborSearch&& other);
   
     ~NeighborSearch();
   
     void Train(MatType referenceSet);
   
     void Train(Tree referenceTree);
   
     void Search(const MatType& querySet,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     void Search(Tree& queryTree,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances,
                 bool sameSet = false);
   
     void Search(const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     static double EffectiveError(arma::mat& foundDistances,
                                  arma::mat& realDistances);
   
     static double Recall(arma::Mat<size_t>& foundNeighbors,
                          arma::Mat<size_t>& realNeighbors);
   
     size_t BaseCases() const { return baseCases; }
   
     size_t Scores() const { return scores; }
   
     NeighborSearchMode SearchMode() const { return searchMode; }
     NeighborSearchMode& SearchMode() { return searchMode; }
   
     double Epsilon() const { return epsilon; }
     double& Epsilon() { return epsilon; }
   
     const MatType& ReferenceSet() const { return *referenceSet; }
   
     const Tree& ReferenceTree() const { return *referenceTree; }
     Tree& ReferenceTree() { return *referenceTree; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     std::vector<size_t> oldFromNewReferences;
     Tree* referenceTree;
     const MatType* referenceSet;
   
     NeighborSearchMode searchMode;
     double epsilon;
   
     MetricType metric;
   
     size_t baseCases;
     size_t scores;
   
     bool treeNeedsReset;
   
     friend class LeafSizeNSWrapper<SortPolicy, TreeType, DualTreeTraversalType,
         SingleTreeTraversalType>;
   }; // class NeighborSearch
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "neighbor_search_impl.hpp"
   
   // Include convenience typedefs.
   #include "typedef.hpp"
   
   #endif
