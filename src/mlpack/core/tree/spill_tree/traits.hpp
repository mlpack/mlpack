/**
 * @file traits.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Specialization of the TreeTraits class for the SpillTree type of tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_TRAITS_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {
namespace tree {

/**
 * This is a specialization of the TreeType class to the SpillTree tree type.
 * It defines characteristics of the spill tree, and is used to help write
 * tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
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
  /**
   * Each spill tree node has two children which can share points.
   * Therefore, children can be overlapping.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * There is no guarantee that the first point in a node is its centroid.
   */
  static const bool FirstPointIsCentroid = false;

  /**
   * Points are not contained at multiple levels of the spill tree.
   */
  static const bool HasSelfChildren = false;

  /**
   * Points are not rearranged during building of the tree.
   */
  static const bool RearrangesDataset = false;

  /**
   * This is always a binary tree.
   */
  static const bool BinaryTree = true;

  /**
   * Spill trees have duplicated points, so NumDescendants() could count a given
   * point twice.
   */
  static const bool UniqueNumDescendants = false;
};

} // namespace tree
} // namespace mlpack

#endif
