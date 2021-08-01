
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_dual_tree_traverser.hpp:

Program Listing for File dual_tree_traverser.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_dual_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/dual_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "binary_space_tree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename BoundMetricType, typename...> class BoundType,
            template<typename SplitBoundType, typename SplitMatType>
                class SplitType>
   template<typename RuleType>
   class BinarySpaceTree<MetricType, StatisticType, MatType, BoundType,
                         SplitType>::DualTreeTraverser
   {
    public:
     DualTreeTraverser(RuleType& rule);
   
     void Traverse(BinarySpaceTree& queryNode,
                   BinarySpaceTree& referenceNode);
   
     size_t NumPrunes() const { return numPrunes; }
     size_t& NumPrunes() { return numPrunes; }
   
     size_t NumVisited() const { return numVisited; }
     size_t& NumVisited() { return numVisited; }
   
     size_t NumScores() const { return numScores; }
     size_t& NumScores() { return numScores; }
   
     size_t NumBaseCases() const { return numBaseCases; }
     size_t& NumBaseCases() { return numBaseCases; }
   
    private:
     RuleType& rule;
   
     size_t numPrunes;
   
     size_t numVisited;
   
     size_t numScores;
   
     size_t numBaseCases;
   
     typename RuleType::TraversalInfoType traversalInfo;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "dual_tree_traverser_impl.hpp"
   
   #endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP
