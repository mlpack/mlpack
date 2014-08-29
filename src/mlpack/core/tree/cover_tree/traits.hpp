/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * This file contains the specialization of the TreeTraits class for the
 * CoverTree type of tree.
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
         typename RootPointPolicy,
         typename StatisticType>
class TreeTraits<CoverTree<MetricType, RootPointPolicy, StatisticType> >
{
 public:
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
};

}; // namespace tree
}; // namespace mlpack

#endif
