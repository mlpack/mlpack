
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_greedy_single_tree_traverser.hpp:

Program Listing for File greedy_single_tree_traverser.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_greedy_single_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/greedy_single_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_GREEDY_SINGLE_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_GREEDY_SINGLE_TREE_TRAVERSER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename TreeType, typename RuleType>
   class GreedySingleTreeTraverser
   {
    public:
     GreedySingleTreeTraverser(RuleType& rule);
   
     void Traverse(const size_t queryIndex, TreeType& referenceNode);
   
     size_t NumPrunes() const { return numPrunes; }
   
    private:
     RuleType& rule;
   
     size_t numPrunes;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "greedy_single_tree_traverser_impl.hpp"
   
   #endif
