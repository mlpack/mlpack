/**
 * @file single_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the SingleTreeTraverser for the cover tree.  This implements a
 * single-tree breadth-first recursion with a pruning rule and a base case (two
 * point) rule.
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
#ifndef __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>

#include "cover_tree.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
class CoverTree<MetricType, RootPointPolicy, StatisticType>::SingleTreeTraverser
{
 public:
  /**
   * Initialize the single tree traverser with the given rule.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is used as
   *      the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, CoverTree& referenceNode);

  //! Get the number of prunes so far.
  size_t NumPrunes() const { return numPrunes; }
  //! Set the number of prunes (good for a reset to 0).
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  //! The number of nodes which have been pruned during traversal.
  size_t numPrunes;
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif
