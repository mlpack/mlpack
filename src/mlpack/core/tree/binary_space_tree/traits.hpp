/**
 * @file core/tree/binary_space_tree/traits.hpp
 * @author Ryan Curtin
 *
 * Specialization of the TreeTraits class for the BinarySpaceTree type of tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>
#include <mlpack/core/tree/ballbound.hpp>

namespace mlpack {

// Utility struct: random projection trees allow overlapping children, so we
// want to capture that as a compile-time constant.
template<template<typename BoundType, typename ElemType> class SplitType>
struct SplitIsOverlapping { static const bool value = false; };

template<>
struct SplitIsOverlapping<RPTreeMaxSplit> { static const bool value = true; };

template<>
struct SplitIsOverlapping<RPTreeMeanSplit> { static const bool value = true; };

// Utility struct: ball bounds, hollow ball bounds, and cell bounds correspond
// to overlapping regions, and we want to capture that as a compile-time
// constant.
template<template<typename DistanceType,
                  typename ElemType,
                  typename...> class BoundType>
struct BoundIsOverlapping { static const bool value = false; };

template<>
struct BoundIsOverlapping<BallBound> { static const bool value = true; };

template<>
struct BoundIsOverlapping<HollowBallBound> { static const bool value = true; };

template<>
struct BoundIsOverlapping<CellBound> { static const bool value = true; };

/**
 * This is a specialization of the TreeTraits class to the BinarySpaceTree tree
 * type.  It defines characteristics of the binary space tree, and is used to
 * help write tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename... BoundExtraParams> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
class TreeTraits<BinarySpaceTree<
    DistanceType, StatisticType, MatType, BoundType, SplitType>>
{
 public:
  /**
   * Each binary space tree node has two children which represent
   * non-overlapping subsets of the space which the node represents.  Therefore,
   * children are not overlapping.
   *
   * There are exceptions: random projection trees allow overlapping children,
   * and the use of ball bounds or cell bounds means that overlapping children
   * are possible.
   */
  static const bool HasOverlappingChildren =
      SplitIsOverlapping<SplitType>::value ||
      BoundIsOverlapping<BoundType>::value;

  /**
   * Each binary space tree node doesn't share points with any other node.
   */
  static const bool HasDuplicatedPoints = false;

  /**
   * There is no guarantee that the first point in a node is its centroid.
   */
  static const bool FirstPointIsCentroid = false;

  /**
   * Points are not contained at multiple levels of the binary space tree.
   */
  static const bool HasSelfChildren = false;

  /**
   * Points are rearranged during building of the tree.
   */
  static const bool RearrangesDataset = true;

  /**
   * This is always a binary tree.
   */
  static const bool BinaryTree = true;

  /**
   * Binary space trees don't have duplicated points, so NumDescendants()
   * represents the number of unique descendant points.
   */
  static const bool UniqueNumDescendants = true;
};

} // namespace mlpack

#endif
