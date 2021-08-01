
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_is_spill_tree.hpp:

Program Listing for File is_spill_tree.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_is_spill_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/spill_tree/is_spill_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_IS_SPILL_TREE_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_IS_SPILL_TREE_HPP
   
   #include "spill_tree.hpp"
   
   namespace mlpack {
   namespace tree  {
   
   // Useful struct when specific behaviour for SpillTrees is required.
   template<typename TreeType>
   struct IsSpillTree
   {
     static const bool value = false;
   };
   
   // Specialization for SpillTree.
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename HyperplaneMetricType>
               class HyperplaneType,
            template<typename SplitMetricType, typename SplitMatType>
               class SplitType>
   struct IsSpillTree<tree::SpillTree<MetricType, StatisticType, MatType,
       HyperplaneType, SplitType>>
   {
     static const bool value = true;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
