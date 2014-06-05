/**
  * @file rectangle_tree_traverser.hpp
  * @author Andrew Wells
  *
  * A class for traversing rectangle type trees with a given set of rules
  * which indicate the branches to prune and the order in which to recurse.
  * This is a depth-first traverser.
  */
#ifndef __MLPAC_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_TRAVERSER_IMPL_HPP
#define __MLPAC_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_TRAVERSER_IMPL_HPP

#include "rectangle_tree_traverser.hpp"

#include <stack>

namespace mlpack {
namespace tree {

template<typename StatisticType,
    typename MatType,
    typename SplitType,
    typename DescentType>
template<typename RuleType>
RectangleTree<StatisticType, MatType, SplitType, DescentType>::
RectangleTreeTraverser<RuleType>::RectangleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do */ }

template<typename StatisticType,
    typename MatType,
    typename SplitType
    typename DescentType>
template<typename RuleType>
void RectangleTree<StatisticType, MatType, SplitType, DescentType>::
RectangleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    RectangeTree<StatisticType, MatyType, SplitType, DescentType>&
        referenceNode)
{

}

}; // namespace tree
}; // namespace mlpack

#endif