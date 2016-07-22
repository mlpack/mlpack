/**
 * @file traits.hpp
 *
 * Specialization of the TreeTraits class for the VantagePointTree type of tree.
 */
#ifndef MLPACK_CORE_TREE_VANTAGE_POINT_TREE_TRAITS_HPP
#define MLPACK_CORE_TREE_VANTAGE_POINT_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {
namespace tree {

/**
 * This is a specialization of the TreeType class to the VantagePointTree tree
 * type.  It defines characteristics of the vantage point tree, and is used to
 * help write tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType, size_t...>
             class SplitType>
class TreeTraits<VantagePointTree<MetricType, StatisticType, MatType, BoundType,
                                 SplitType>>
{
 public:
  /**
   * Children nodes may overlap each other.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * Returns true if the first point of the node is the centroid of its bound.
   *
   * @param node The node to check.
   */
  static bool FirstPointIsCentroid(const VantagePointTree<MetricType,
      StatisticType, MatType, BoundType, SplitType>*  node)
  {
    return node->FirstPointIsCentroid();
  }

  /**
   * Points are not contained at multiple levels of the vantage point tree.
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
};

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_TRAITS_HPP

