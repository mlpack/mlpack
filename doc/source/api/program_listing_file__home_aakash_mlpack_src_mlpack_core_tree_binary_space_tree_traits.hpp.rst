
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_traits.hpp:

Program Listing for File traits.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP
   
   #include <mlpack/core/tree/tree_traits.hpp>
   #include <mlpack/core/tree/ballbound.hpp>
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename BoundMetricType, typename...> class BoundType,
            template<typename SplitBoundType, typename SplitMatType>
                class SplitType>
   class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType, BoundType,
                                    SplitType>>
   {
    public:
     static const bool HasOverlappingChildren = false;
   
     static const bool HasDuplicatedPoints = false;
   
     static const bool FirstPointIsCentroid = false;
   
     static const bool HasSelfChildren = false;
   
     static const bool RearrangesDataset = true;
   
     static const bool BinaryTree = true;
   
     static const bool UniqueNumDescendants = true;
   };
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename BoundMetricType, typename...> class BoundType>
   class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType, BoundType,
                                    RPTreeMaxSplit>>
   {
    public:
     static const bool HasOverlappingChildren = true;
   
     static const bool HasDuplicatedPoints = false;
   
     static const bool FirstPointIsCentroid = false;
   
     static const bool HasSelfChildren = false;
   
     static const bool RearrangesDataset = true;
   
     static const bool BinaryTree = true;
   
     static const bool UniqueNumDescendants = true;
   };
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename BoundMetricType, typename...> class BoundType>
   class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType, BoundType,
                                    RPTreeMeanSplit>>
   {
    public:
     static const bool HasOverlappingChildren = true;
   
     static const bool HasDuplicatedPoints = false;
   
     static const bool FirstPointIsCentroid = false;
   
     static const bool HasSelfChildren = false;
   
     static const bool RearrangesDataset = true;
   
     static const bool BinaryTree = true;
   
     static const bool UniqueNumDescendants = true;
   };
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename SplitBoundType, typename SplitMatType>
                class SplitType>
   class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType,
       bound::BallBound, SplitType>>
   {
    public:
     static const bool HasOverlappingChildren = true;
     static const bool HasDuplicatedPoints = false;
     static const bool FirstPointIsCentroid = false;
     static const bool HasSelfChildren = false;
     static const bool RearrangesDataset = true;
     static const bool BinaryTree = true;
     static const bool UniqueNumDescendants = true;
   };
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename SplitBoundType, typename SplitMatType>
                class SplitType>
   class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType,
       bound::HollowBallBound, SplitType>>
   {
    public:
     static const bool HasOverlappingChildren = true;
     static const bool HasDuplicatedPoints = false;
     static const bool FirstPointIsCentroid = false;
     static const bool HasSelfChildren = false;
     static const bool RearrangesDataset = true;
     static const bool BinaryTree = true;
     static const bool UniqueNumDescendants = true;
   };
   
   template<typename MetricType,
            typename StatisticType,
            typename MatType,
            template<typename SplitBoundType, typename SplitMatType>
                class SplitType>
   class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType,
       bound::CellBound, SplitType>>
   {
    public:
     static const bool HasOverlappingChildren = true;
     static const bool HasDuplicatedPoints = false;
     static const bool FirstPointIsCentroid = false;
     static const bool HasSelfChildren = false;
     static const bool RearrangesDataset = true;
     static const bool BinaryTree = true;
     static const bool UniqueNumDescendants = true;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
