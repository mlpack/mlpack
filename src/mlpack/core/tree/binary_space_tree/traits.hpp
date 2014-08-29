/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * Specialization of the TreeTraits class for the BinarySpaceTree type of tree.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_TRAITS_HPP

#include <mlpack/core/tree/tree_traits.hpp>

namespace mlpack {
namespace tree {

/**
 * This is a specialization of the TreeType class to the BinarySpaceTree tree
 * type.  It defines characteristics of the binary space tree, and is used to
 * help write tree-independent (but still optimized) tree-based algorithms.  See
 * mlpack/core/tree/tree_traits.hpp for more information.
 */
template<typename BoundType,
         typename StatisticType,
         typename MatType>
class TreeTraits<BinarySpaceTree<BoundType, StatisticType, MatType> >
{
 public:
  /**
   * Each binary space tree node has two children which represent
   * non-overlapping subsets of the space which the node represents.  Therefore,
   * children are not overlapping.
   */
  static const bool HasOverlappingChildren = false;

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
};

}; // namespace tree
}; // namespace mlpack

#endif
