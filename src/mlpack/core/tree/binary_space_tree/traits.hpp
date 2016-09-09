/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * Specialization of the TreeTraits class for the BinarySpaceTree type of tree.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>
#include <mlpack/core/tree/ballbound.hpp>

namespace mlpack {
namespace tree {

/**
 * This is a specialization of the TreeType class to the BinarySpaceTree tree
 * type.  It defines characteristics of the binary space tree, and is used to
 * help write tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
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
  /**
   * Each binary space tree node has two children which represent
   * non-overlapping subsets of the space which the node represents.  Therefore,
   * children are not overlapping.
   */
  static const bool HasOverlappingChildren = false;

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

/**
 * This is a specialization of the TreeType class to the max-split random
 * projection tree. The only difference with general BinarySpaceTree is that the
 * tree can have overlapping children.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType>
class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType, BoundType,
                                 RPTreeMaxSplit>>
{
 public:
  /**
   * Children of a random projection tree node may overlap.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * The tree has not got duplicated points.
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

/**
 * This is a specialization of the TreeType class to the mean-split random
 * projection tree. The only difference with general BinarySpaceTree is that the
 * tree can have overlapping children.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType>
class TreeTraits<BinarySpaceTree<MetricType, StatisticType, MatType, BoundType,
                                 RPTreeMeanSplit>>
{
 public:
  /**
   * Children of a random projection tree node may overlap.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * The tree has not got duplicated points.
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

/**
 * This is a specialization of the TreeType class to the BallTree tree type.
 * The only difference with general BinarySpaceTree is that BallTree can have
 * overlapping children.
 * See mlpack/core/tree/tree_traits.hpp for more information.
 */
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

/**
 * This is a specialization of the TreeType class to an arbitrary tree with
 * HollowBallBound (currently only the vantage point tree is supported).
 * The only difference with general BinarySpaceTree is that the tree can have
 * overlapping children.
 */
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

/**
 * This is a specialization of the TreeType class to the UBTree tree type.
 * The only difference with general BinarySpaceTree is that UBTree can have
 * overlapping children.
 * See mlpack/core/tree/tree_traits.hpp for more information.
 */
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
