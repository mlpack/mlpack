/**
 * @file traits.hpp
 * @author Andrew Wells
 *
 * Specialization of the TreeTraits class for the RectangleTree type of tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_TRAITS_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {
namespace tree {

/**
 * This is a specialization of the TreeType class to the RectangleTree tree
 * type.  It defines characteristics of the rectangle type trees, and is used to
 * help write tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
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
  /**
   * An R-tree can have overlapping children.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * An R-tree node doesn't share points with another node.
   */
  static const bool HasDuplicatedPoints = false;

  /**
   * There is no guarantee that the first point in a node is its centroid.
   */
  static const bool FirstPointIsCentroid = false;

  /**
   * Points are not contained at multiple levels of the R-tree.
   */
  static const bool HasSelfChildren = false;

  /**
   * Points are rearranged during building of the tree.
   * THIS MAY NOT BE TRUE.  IT'S HARD TO DYNAMICALLY INSERT POINTS
   * AND REARRANGE THE MATRIX
   */
  static const bool RearrangesDataset = false;

  /**
   * This tree is not necessarily a binary tree.
   */
  static const bool BinaryTree = false;

  /**
   * Rectangle trees don't have duplicated points, so NumDescendants()
   * represents the number of unique descendant points.
   */
  static const bool UniqueNumDescendants = true;
};

/**
 * Since the R+/R++ tree can not have overlapping children, we should define
 * traits for the R+/R++ tree.
 */
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
  /**
   * The R+/R++ tree can't have overlapping children.
   */
  static const bool HasOverlappingChildren = false;

  /**
   * An R-tree node doesn't share points with another node.
   */
  static const bool HasDuplicatedPoints = false;

  /**
   * There is no guarantee that the first point in a node is its centroid.
   */
  static const bool FirstPointIsCentroid = false;

  /**
   * Points are not contained at multiple levels of the R-tree.
   */
  static const bool HasSelfChildren = false;

  /**
   * Points are rearranged during building of the tree.
   * THIS MAY NOT BE TRUE.  IT'S HARD TO DYNAMICALLY INSERT POINTS
   * AND REARRANGE THE MATRIX
   */
  static const bool RearrangesDataset = false;

  /**
   * This tree is not necessarily a binary tree.
   */
  static const bool BinaryTree = false;

  /**
   * Rectangle trees don't have duplicated points, so NumDescendants()
   * represents the number of unique descendant points.
   */
  static const bool UniqueNumDescendants = true;
};

} // namespace tree
} // namespace mlpack

#endif
