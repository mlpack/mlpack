
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_typedef.hpp:

Program Listing for File typedef.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_typedef.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/typedef.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_TYPEDEF_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_TYPEDEF_HPP
   
   #include "rectangle_tree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using RTree = RectangleTree<MetricType,
                               StatisticType,
                               MatType,
                               RTreeSplit,
                               RTreeDescentHeuristic,
                               NoAuxiliaryInformation>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using RStarTree = RectangleTree<MetricType,
                                   StatisticType,
                                   MatType,
                                   RStarTreeSplit,
                                   RStarTreeDescentHeuristic,
                                   NoAuxiliaryInformation>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using XTree = RectangleTree<MetricType,
                               StatisticType,
                               MatType,
                               XTreeSplit,
                               RTreeDescentHeuristic,
                               XTreeAuxiliaryInformation>;
   
   template<typename TreeType>
   using DiscreteHilbertRTreeAuxiliaryInformation =
         HilbertRTreeAuxiliaryInformation<TreeType, DiscreteHilbertValue>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using HilbertRTree = RectangleTree<MetricType,
                               StatisticType,
                               MatType,
                               HilbertRTreeSplit<2>,
                               HilbertRTreeDescentHeuristic,
                               DiscreteHilbertRTreeAuxiliaryInformation>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using RPlusTree = RectangleTree<MetricType,
                               StatisticType,
                               MatType,
                               RPlusTreeSplit<RPlusTreeSplitPolicy,
                                              MinimalCoverageSweep>,
                               RPlusTreeDescentHeuristic,
                               NoAuxiliaryInformation>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using RPlusPlusTree = RectangleTree<MetricType,
                               StatisticType,
                               MatType,
                               RPlusTreeSplit<RPlusPlusTreeSplitPolicy,
                                              MinimalSplitsNumberSweep>,
                               RPlusPlusTreeDescentHeuristic,
                               RPlusPlusTreeAuxiliaryInformation>;
   } // namespace tree
   } // namespace mlpack
   
   #endif
