/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * This file contains the specialization of the TreeTraits class for the
 * CoverTree type of tree.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_TRAITS_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {
namespace tree {

/**
 * The specialization of the TreeTraits class for the CoverTree tree type.  It
 * defines characteristics of the cover tree, and is used to help write
 * tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
struct TreeTraits<CoverTree<MetricType, StatisticType, MatType,
                            RootPointPolicy>>
{
  /**
   * The cover tree (or, this implementation of it) does not require that
   * children represent non-overlapping subsets of the parent node.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * Each cover tree node contains only one point, and that point is its
   * centroid.
   */
  static const bool FirstPointIsCentroid = true;

  /**
   * Cover trees do have self-children.
   */
  static const bool HasSelfChildren = true;

  /**
   * Points are not rearranged when the tree is built.
   */
  static const bool RearrangesDataset = false;

  /**
   * The cover tree is not necessarily a binary tree.
   */
  static const bool BinaryTree = false;
};

} // namespace tree
} // namespace mlpack

#endif
