/**
 * @file dual_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the DualTreeTraverser for the BinarySpaceTree tree type.  This is a
 * nested class of BinarySpaceTree which traverses two trees in a depth-first
 * manner with a given set of rules which indicate the branches which can be
 * pruned and the order in which to recurse.
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
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>

#include "binary_space_tree.hpp"

namespace mlpack {
namespace tree {

template<typename BoundType,
         typename StatisticType,
         typename MatType,
         typename SplitType>
template<typename RuleType>
class BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>::
    DualTreeTraverser
{
 public:
  /**
   * Instantiate the dual-tree traverser with the given rule set.
   */
  DualTreeTraverser(RuleType& rule);

  /**
   * Traverse the two trees.  This does not reset the number of prunes.
   *
   * @param queryNode The query node to be traversed.
   * @param referenceNode The reference node to be traversed.
   * @param score The score of the current node combination.
   */
  void Traverse(BinarySpaceTree& queryNode,
                BinarySpaceTree& referenceNode);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of prunes.
  size_t& NumPrunes() { return numPrunes; }

  //! Get the number of visited combinations.
  size_t NumVisited() const { return numVisited; }
  //! Modify the number of visited combinations.
  size_t& NumVisited() { return numVisited; }

  //! Get the number of times a node combination was scored.
  size_t NumScores() const { return numScores; }
  //! Modify the number of times a node combination was scored.
  size_t& NumScores() { return numScores; }

  //! Get the number of times a base case was calculated.
  size_t NumBaseCases() const { return numBaseCases; }
  //! Modify the number of times a base case was calculated.
  size_t& NumBaseCases() { return numBaseCases; }

 private:
  //! Reference to the rules with which the trees will be traversed.
  RuleType& rule;

  //! The number of prunes.
  size_t numPrunes;

  //! The number of node combinations that have been visited during traversal.
  size_t numVisited;

  //! The number of times a node combination was scored.
  size_t numScores;

  //! The number of times a base case was calculated.
  size_t numBaseCases;

  //! Traversal information, held in the class so that it isn't continually
  //! being reallocated.
  typename RuleType::TraversalInfoType traversalInfo;
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "dual_tree_traverser_impl.hpp"

#endif // __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP

