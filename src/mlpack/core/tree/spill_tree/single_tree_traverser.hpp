/**
 * @file single_tree_traverser.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * A nested class of SpillTree which traverses the entire tree with a
 * given set of rules which indicate the branches which can be pruned and the
 * order in which to recurse.  This traverser is a depth-first traverser.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>

#include "spill_tree.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
class SpillTree<MetricType, StatisticType, MatType, SplitType>::
    SingleTreeTraverser
{
 public:
  /**
   * Instantiate the single tree traverser with the given rule set.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is being
   *     used as the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, SpillTree& referenceNode);

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

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif
