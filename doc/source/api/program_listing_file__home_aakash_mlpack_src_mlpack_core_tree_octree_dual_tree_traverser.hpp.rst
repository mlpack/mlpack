
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_octree_dual_tree_traverser.hpp:

Program Listing for File dual_tree_traverser.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_octree_dual_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/octree/dual_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_OCTREE_DUAL_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_OCTREE_DUAL_TREE_TRAVERSER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "octree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType>
   template<typename RuleType>
   class Octree<MetricType, StatisticType, MatType>::DualTreeTraverser
   {
    public:
     DualTreeTraverser(RuleType& rule);
   
     void Traverse(Octree& queryNode, Octree& referenceNode);
   
     size_t NumPrunes() const { return numPrunes; }
     size_t& NumPrunes() { return numPrunes; }
   
     size_t NumVisited() const { return numVisited; }
     size_t& NumVistied() { return numVisited; }
   
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
   
   #endif
