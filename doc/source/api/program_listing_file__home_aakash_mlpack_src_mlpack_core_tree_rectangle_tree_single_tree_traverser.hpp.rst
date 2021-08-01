
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_single_tree_traverser.hpp:

Program Listing for File single_tree_traverser.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_single_tree_traverser.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/single_tree_traverser.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_SINGLE_TREE_TRAVERSER_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_SINGLE_TREE_TRAVERSER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "rectangle_tree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            typename SplitType,
            typename DescentType,
            template<typename> class AuxiliaryInformationType>
   template<typename RuleType>
   class RectangleTree<MetricType, StatisticType, MatType, SplitType,
                       DescentType, AuxiliaryInformationType>::SingleTreeTraverser
   {
    public:
     SingleTreeTraverser(RuleType& rule);
   
     void Traverse(const size_t queryIndex, const RectangleTree& referenceNode);
   
     size_t NumPrunes() const { return numPrunes; }
     size_t& NumPrunes() { return numPrunes; }
   
    private:
     // We use this class and this function to make the sorting and scoring easy
     // and efficient:
     struct NodeAndScore
     {
       RectangleTree* node;
       double score;
     };
   
     static bool NodeComparator(const NodeAndScore& obj1, const NodeAndScore& obj2)
     {
       return obj1.score < obj2.score;
     }
   
     RuleType& rule;
   
     size_t numPrunes;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   // Include implementation.
   #include "single_tree_traverser_impl.hpp"
   
   #endif
