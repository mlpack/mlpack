
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_single_tree_traverser.hpp:

Program Listing for File single_tree_traverser.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_single_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/single_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_SINGLE_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_SINGLE_TREE_TRAVERSER_HPP
   
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
                         SplitType>::SingleTreeTraverser
   {
    public:
     SingleTreeTraverser(RuleType& rule);
   
     void Traverse(const size_t queryIndex, BinarySpaceTree& referenceNode);
   
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
