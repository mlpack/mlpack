
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_dual_tree_traverser.hpp:

Program Listing for File dual_tree_traverser.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_cover_tree_dual_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/cover_tree/dual_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <queue>
   
   namespace mlpack {
   namespace tree {
   
   template<
       typename MetricType,
       typename StatisticType,
       typename MatType,
       typename RootPointPolicy
   >
   template<typename RuleType>
   class CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
       DualTreeTraverser
   {
    public:
     DualTreeTraverser(RuleType& rule);
   
     void Traverse(CoverTree& queryNode, CoverTree& referenceNode);
   
     size_t NumPrunes() const { return numPrunes; }
     size_t& NumPrunes() { return numPrunes; }
   
     size_t NumVisited() const { return 0; }
     size_t NumScores() const { return 0; }
     size_t NumBaseCases() const { return 0; }
   
    private:
     RuleType& rule;
   
     size_t numPrunes;
   
     struct DualCoverTreeMapEntry
     {
       CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>*
           referenceNode;
       double score;
       double baseCase;
       typename RuleType::TraversalInfoType traversalInfo;
   
       bool operator<(const DualCoverTreeMapEntry& other) const
       {
         if (score == other.score)
           return (baseCase < other.baseCase);
         else
           return (score < other.score);
       }
     };
   
     void Traverse(
         CoverTree& queryNode,
         std::map<int, std::vector<DualCoverTreeMapEntry>,
             std::greater<int>>& referenceMap);
   
     void PruneMap(
         CoverTree& queryNode,
         std::map<int, std::vector<DualCoverTreeMapEntry>,
             std::greater<int>>& referenceMap,
         std::map<int, std::vector<DualCoverTreeMapEntry>,
             std::greater<int>>& childMap);
   
     void ReferenceRecursion(
         CoverTree& queryNode,
       std::map<int, std::vector<DualCoverTreeMapEntry>,
           std::greater<int>>& referenceMap);
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "dual_tree_traverser_impl.hpp"
   
   #endif
