/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Implementation of the DualTreeTraverser for SpillTree.  This is a way
 * to perform a dual-tree traversal of two trees.  The trees must be the same
 * type.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
SpillTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do. */ }

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType>
             class SplitType>
template<typename RuleType>
void SpillTree<MetricType, StatisticType, MatType, BoundType, SplitType>::
DualTreeTraverser<RuleType>::Traverse(
    SpillTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        /* queryNode */,
    SpillTree<MetricType, StatisticType, MatType, BoundType, SplitType>&
        /* referenceNode */)
{
  // TODO: Add support for dual tree traverser.
  throw std::runtime_error("Dual tree traverser not implemented for "
      "spill trees.");
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_SPILL_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
