/**
  * @file rectangle_tree_traverser_impl.hpp
  * @author Andrew Wells
  *
  * A class for traversing rectangle type trees with a given set of rules
  * which indicate the branches to prune and the order in which to recurse.
  * This is a depth-first traverser.
  */
#ifndef __MLPAC_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_TRAVERSER_IMPL_HPP
#define __MLPAC_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_TRAVERSER_IMPL_HPP

#include "rectangle_tree_traverser.hpp"

#include <algorithm>
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
  // If we reach a leaf node, we need to run the base case.
  if(referenceNode.IsLeaf()) {
    for(size_t i = 0; i < referenceNode.Count(); i++) {
      rule.BaseCase(queryIndex, i);
    }
    return;
  }
  
  // This is not a leaf node so we:
  // Sort the children of this node by their scores.
  std::vector<RectangleTree*> nodes = new std::vector<RectangleTree*>(referenceNode.NumChildren());
  std::vector<double> scores = new std::vector<double>(referenceNode.NumChildren());
  for(int i = 0; i < referenceNode.NumChildren(); i++) {
    nodes[i] = referenceNode.Children()[i];
    scores[i] = Rule.Score(nodes[i]);
  }  
  Rule.sortNodesAndScores(&nodes, &scores);
  
  // Iterate through them starting with the best and stopping when we reach
  // one that isn't good enough.
  for(int i = 0; i < referenceNode.NumChildren(); i++) {
   if(Rule.Rescore(queryIndex, nodes[i], scores[i]) != DBL_MAX)
     Traverse(queryIndex, nodes[i]);
   else {
    numPrunes += referenceNode.NumChildren - i;
    return;
   }
  }
  // We only get here if we couldn't prune any of them.
  return;
}

}; // namespace tree
}; // namespace mlpack

#endif