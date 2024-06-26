/**
 * @file core/tree/spill_tree/spill_single_tree_traverser.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * A nested class of SpillTree which traverses the entire tree with a
 * given set of rules which indicate the branches which can be pruned and the
 * order in which to recurse.  This traverser is a depth-first traverser.
 * The Defeatist template parameter determines if the traversers must do
 * defeatist search on overlapping nodes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPILL_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPILL_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>

#include "spill_tree.hpp"

namespace mlpack {

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneDistanceType> class HyperplaneType,
         template<typename SplitDistanceType, typename SplitMatType>
             class SplitType>
template<typename RuleType, bool Defeatist>
class SpillTree<DistanceType, StatisticType, MatType, HyperplaneType,
                SplitType>::SpillSingleTreeTraverser
{
 public:
  /**
   * Instantiate the single tree traverser with the given rule set.
   */
  SpillSingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is being
   *     used as the query point.
   * @param referenceNode The tree node to be traversed.
   * @param bruteForce If true, then do a brute-force search on the reference
   *     node instead of traversing any further.
   */
  void Traverse(const size_t queryIndex,
                SpillTree& referenceNode,
                const bool bruteForce = false);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of prunes.
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  //! The number of nodes which have been pruned during traversal.
  size_t numPrunes;
};

} // namespace mlpack

// Include implementation.
#include "spill_single_tree_traverser_impl.hpp"

#endif
