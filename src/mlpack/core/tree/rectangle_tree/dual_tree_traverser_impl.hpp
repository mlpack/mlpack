/**
  * @file dual_tree_traverser_impl.hpp
  * @author Andrew Wells
  *
  * A class for traversing rectangle type trees with a given set of rules
  * which indicate the branches to prune and the order in which to recurse.
  * This is a depth-first traverser.
  */
#ifndef __MLPAC_CORE_TREE_RECTANGLE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define __MLPAC_CORE_TREE_RECTANGLE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

#include "dual_tree_traverser.hpp"

#include <algorithm>
#include <stack>

namespace mlpack {
namespace tree {

template<typename SplitType,
         typename DescentType,
	 typename StatisticType,
         typename MatType>
template<typename RuleType>
RectangleTree<SplitType, DescentType, StatisticType, MatType>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do */ }

template<typename SplitType,
         typename DescentType,
	 typename StatisticType,
         typename MatType>
template<typename RuleType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::
DualTreeTraverser<RuleType>::Traverse(RectangleTree<SplitType, DescentType, StatisticType, MatType>& queryNode,
		RectangleTree<SplitType, DescentType, StatisticType, MatType>& referenceNode)
{
  //Do nothing.  Just here to prevent warnings.
  if(queryNode.NumDescendants() > referenceNode.NumDescendants())
    return;
  return;
}

}; // namespace tree
}; // namespace mlpack

#endif