
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_traits.hpp:

Program Listing for File traits.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_rectangle_tree_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/rectangle_tree/traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_TRAITS_HPP
   #define MLPACK_CORE_TREE_RECTANGLE_TREE_TRAITS_HPP
   
   #include <mlpack/core/tree/tree_traits.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            typename SplitType,
            typename DescentType,
            template<typename> class AuxiliaryInformationType>
   class TreeTraits<RectangleTree<MetricType, StatisticType, MatType, SplitType,
                                  DescentType, AuxiliaryInformationType>>
   {
    public:
     static const bool HasOverlappingChildren = true;
   
     static const bool HasDuplicatedPoints = false;
   
     static const bool FirstPointIsCentroid = false;
   
     static const bool HasSelfChildren = false;
   
     static const bool RearrangesDataset = false;
   
     static const bool BinaryTree = false;
   
     static const bool UniqueNumDescendants = true;
   };
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            typename SplitPolicyType,
            template<typename> class SweepType,
            typename DescentType,
            template<typename> class AuxiliaryInformationType>
   class TreeTraits<RectangleTree<MetricType,
       StatisticType,
       MatType,
       RPlusTreeSplit<SplitPolicyType,
                      SweepType>,
       DescentType,
       AuxiliaryInformationType>>
   {
    public:
     static const bool HasOverlappingChildren = false;
   
     static const bool HasDuplicatedPoints = false;
   
     static const bool FirstPointIsCentroid = false;
   
     static const bool HasSelfChildren = false;
   
     static const bool RearrangesDataset = false;
   
     static const bool BinaryTree = false;
   
     static const bool UniqueNumDescendants = true;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
