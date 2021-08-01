
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_spill_single_tree_traverser.hpp:

Program Listing for File spill_single_tree_traverser.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_spill_single_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/spill_tree/spill_single_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_SPILL_SINGLE_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_SPILL_SINGLE_TREE_TRAVERSER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "spill_tree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename HyperplaneMetricType> class HyperplaneType,
            template<typename SplitMetricType, typename SplitMatType>
                class SplitType>
   template<typename RuleType, bool Defeatist>
   class SpillTree<MetricType, StatisticType, MatType, HyperplaneType, SplitType>::
       SpillSingleTreeTraverser
   {
    public:
     SpillSingleTreeTraverser(RuleType& rule);
   
     void Traverse(const size_t queryIndex,
                   SpillTree& referenceNode,
                   const bool bruteForce = false);
   
     size_t NumPrunes() const { return numPrunes; }
     size_t& NumPrunes() { return numPrunes; }
   
    private:
     RuleType& rule;
   
     size_t numPrunes;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "spill_single_tree_traverser_impl.hpp"
   
   #endif
