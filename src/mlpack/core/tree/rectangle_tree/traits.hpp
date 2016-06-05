/**
 * @file traits.hpp
 * @author Andrew Wells
 *
 * Specialization of the TreeTraits class for the RectangleTree type of tree.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
         template<typename> class SplitType,
         typename DescentType>
class TreeTraits<RectangleTree<MetricType, StatisticType, MatType, SplitType,
                               DescentType>>
{
 public:
  /**
   * An R-tree can have overlapping children.
   */
  static const bool HasOverlappingChildren = true;

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
};

} // namespace tree
} // namespace mlpack

#endif
