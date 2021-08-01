
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_octree_single_tree_traverser.hpp:

Program Listing for File single_tree_traverser.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_octree_single_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/octree/single_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_OCTREE_SINGLE_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_OCTREE_SINGLE_TREE_TRAVERSER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "octree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType, typename StatisticType, typename MatType>
   template<typename RuleType>
   class Octree<MetricType, StatisticType, MatType>::SingleTreeTraverser
   {
    public:
     SingleTreeTraverser(RuleType& rule);
   
     void Traverse(const size_t queryIndex, Octree& referenceNode);
   
     size_t NumPrunes() const { return numPrunes; }
     size_t& NumPrunes() { return numPrunes; }
   
    private:
     RuleType& rule;
     size_t numPrunes;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "single_tree_traverser_impl.hpp"
   
   #endif
