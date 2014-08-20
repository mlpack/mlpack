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
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do */ }

template<typename SplitType,
         typename DescentType,
	 typename StatisticType,
         typename MatType>
template<typename RuleType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::
DualTreeTraverser<RuleType>::Traverse(
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& queryNode,
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& referenceNode)
{
  // Increment the visit counter.
  ++numVisited;
  
  // Store the current traversal info.
  traversalInfo = rule.TraversalInfo();
  
  // We now have four options.
  // 1)  Both nodes are leaf nodes.
  // 2)  Only the reference node is a leaf node.
  // 3)  Only the query node is a leaf node.
  // 4)  Niether node is a leaf node.
  // We go through those options in that order.
  
  if(queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    // Evaluate the base case.  Do the query points on the outside so we can possibly
    // prune the reference node for that particular point.
    for(size_t query = 0; query < queryNode.Count(); ++query)
    {
      // Restore the traversal information.
      rule.TraversalInfo() = traversalInfo;
      const double childScore = rule.Score(queryNode.Points()[query], referenceNode);
      
      if(childScore == DBL_MAX)
        continue;  // This point doesn't require a search in this reference node.
      
      for(size_t ref = 0; ref < referenceNode.Count(); ++ref)
        rule.BaseCase(queryNode.Points()[query], referenceNode.Points()[ref]);
      
      numBaseCases += referenceNode.Count();
    }
  }
  else if(!queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    // We only need to traverse down the query node.  Order doesn't matter here.
    ++numScores;
    if(rule.Score(queryNode.Child(0), referenceNode) < DBL_MAX)
      Traverse(queryNode.Child(0), referenceNode);
    else
      numPrunes++;
    
    for(size_t i = 1; i < queryNode.NumChildren(); ++i)
    {
      // Before recursing, we have to set the traversal information correctly.
      rule.TraversalInfo() = traversalInfo;
      ++numScores;
      if(rule.Score(queryNode.Child(i), referenceNode) < DBL_MAX)
        Traverse(queryNode.Child(i), referenceNode);
      else
        numPrunes++;
    }       
  }
  else if(queryNode.IsLeaf() && !referenceNode.IsLeaf())
  {
    // We only need to traverse down the reference node.  Order does matter here.
    
    // We sort the children of the reference node by their scores.    
    std::vector<NodeAndScore> nodesAndScores(referenceNode.NumChildren());
    for(int i = 0; i < referenceNode.NumChildren(); i++)
    {
      nodesAndScores[i].node = referenceNode.Children()[i];
      nodesAndScores[i].score = rule.Score(queryNode, *(nodesAndScores[i].node));
    }
    std::sort(nodesAndScores.begin(), nodesAndScores.end(), nodeComparator);
    numScores += nodesAndScores.size();
    
    for(int i = 0; i < nodesAndScores.size(); i++)
    {
      if(rule.Rescore(queryNode, *(nodesAndScores[i].node), nodesAndScores[i].score) < DBL_MAX) {
        rule.TraversalInfo() = traversalInfo;
        Traverse(queryNode, *(nodesAndScores[i].node));
      } else {
        numPrunes += nodesAndScores.size() - i;
        break;
      }
    }
  }
  else
  {
    // We need to traverse down both the reference and the query trees.
    // We loop through all of the query nodes, and for each of them, we
    // loop through the reference nodes to see where we need to descend.
    
    for(int j = 0; j < queryNode.NumChildren(); j++)
    {
      // We sort the children of the reference node by their scores.    
      std::vector<NodeAndScore> nodesAndScores(referenceNode.NumChildren());
      for(int i = 0; i < referenceNode.NumChildren(); i++)
      {
        nodesAndScores[i].node = referenceNode.Children()[i];
        nodesAndScores[i].score = rule.Score(queryNode, *nodesAndScores[i].node);
      }
      std::sort(nodesAndScores.begin(), nodesAndScores.end(), nodeComparator);
      numScores += nodesAndScores.size();
    
      for(int i = 0; i < nodesAndScores.size(); i++)
      {
        if(rule.Rescore(queryNode, *(nodesAndScores[i].node), nodesAndScores[i].score) < DBL_MAX) {
          rule.TraversalInfo() = traversalInfo;
          Traverse(queryNode, *(nodesAndScores[i].node));
        } else {
          numPrunes += nodesAndScores.size() - i;
          break;
        }
      }
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif