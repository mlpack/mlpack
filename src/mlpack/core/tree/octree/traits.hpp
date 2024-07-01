/**
 * @file core/tree/octree/traits.hpp
 * @author Ryan Curtin
 *
 * Specialization of the TreeTraits class for the Octree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_OCTREE_TRAITS_HPP
#define MLPACK_CORE_TREE_OCTREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {

/**
 * This is a specialization of the TreeTraits class to the Octree tree type.  It
 * defines characteristics of the octree, and is used to help write
 * tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType>
class TreeTraits<Octree<DistanceType, StatisticType, MatType>>
{
 public:
  /**
   * No octree nodes will overlap.
   */
  static const bool HasOverlappingChildren = false;

  /**
   * Points are not shared across nodes in the octree.
   */
  static const bool HasDuplicatedPoints = false;

  /**
   * There is no guarantee that the first point in a node is its centroid.
   */
  static const bool FirstPointIsCentroid = false;

  /**
   * Points are not contained at multiple levels of the octree.
   */
  static const bool HasSelfChildren = false;

  /**
   * Points are rearranged during building of the tree.
   */
  static const bool RearrangesDataset = true;

  /**
   * This is not necessarily a binary tree.
   */
  static const bool BinaryTree = false;

  /**
   * NumDescendants() represents the number of unique descendant points.
   */
  static const bool UniqueNumDescendants = true;
};

} // namespace mlpack

#endif
