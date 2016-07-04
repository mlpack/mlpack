/**
 * @file traits.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Specialization of the TreeTraits class for the SpillTree type of tree.
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
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
class TreeTraits<SpillTree<MetricType, StatisticType, MatType, BoundType,
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
};

} // namespace tree
} // namespace mlpack

#endif
