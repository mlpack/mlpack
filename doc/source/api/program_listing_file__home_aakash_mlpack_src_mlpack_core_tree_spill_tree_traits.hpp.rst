
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_traits.hpp:

Program Listing for File traits.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/spill_tree/traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_TRAITS_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_TRAITS_HPP
   
   #include <mlpack/core/tree/tree_traits.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename HyperplaneMetricType> class HyperplaneType,
            template<typename SplitMetricType, typename SplitMatType>
                class SplitType>
   class TreeTraits<SpillTree<MetricType, StatisticType, MatType, HyperplaneType,
       SplitType>>
   {
    public:
     static const bool HasOverlappingChildren = true;
   
     static const bool FirstPointIsCentroid = false;
   
     static const bool HasSelfChildren = false;
   
     static const bool RearrangesDataset = false;
   
     static const bool BinaryTree = true;
   
     static const bool UniqueNumDescendants = false;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
