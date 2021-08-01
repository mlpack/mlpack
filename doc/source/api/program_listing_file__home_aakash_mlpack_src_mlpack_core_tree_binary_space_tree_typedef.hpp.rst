
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_typedef.hpp:

Program Listing for File typedef.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_binary_space_tree_typedef.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/binary_space_tree/typedef.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP
   #define MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP
   
   // In case it hasn't been included yet.
   #include "../binary_space_tree.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using KDTree = BinarySpaceTree<MetricType,
                                  StatisticType,
                                  MatType,
                                  bound::HRectBound,
                                  MidpointSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using MeanSplitKDTree = BinarySpaceTree<MetricType,
                                           StatisticType,
                                           MatType,
                                           bound::HRectBound,
                                           MeanSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using BallTree = BinarySpaceTree<MetricType,
                                    StatisticType,
                                    MatType,
                                    bound::BallBound,
                                    MidpointSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using MeanSplitBallTree = BinarySpaceTree<MetricType,
                                             StatisticType,
                                             MatType,
                                             bound::BallBound,
                                             MeanSplit>;
   
   template<typename BoundType,
            typename MatType = arma::mat>
   using VPTreeSplit = VantagePointSplit<BoundType, MatType, 100>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using VPTree = BinarySpaceTree<MetricType,
                                  StatisticType,
                                  MatType,
                                  bound::HollowBallBound,
                                  VPTreeSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using MaxRPTree = BinarySpaceTree<MetricType,
                                     StatisticType,
                                     MatType,
                                     bound::HRectBound,
                                     RPTreeMaxSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using RPTree = BinarySpaceTree<MetricType,
                                     StatisticType,
                                     MatType,
                                     bound::HRectBound,
                                     RPTreeMeanSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using UBTree = BinarySpaceTree<MetricType,
                                  StatisticType,
                                  MatType,
                                  bound::CellBound,
                                  UBTreeSplit>;
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
