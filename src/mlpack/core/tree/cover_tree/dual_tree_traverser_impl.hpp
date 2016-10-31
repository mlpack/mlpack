/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

#include <mlpack/core.hpp>
#include <queue>

namespace mlpack {
namespace tree {

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::Traverse(CoverTree& queryNode,
                                      CoverTree& referenceNode)
{
  // Start by creating a map and adding the reference root node to it.
  std::map<int, std::vector<DualCoverTreeMapEntry> > refMap;

  DualCoverTreeMapEntry rootRefEntry;

  rootRefEntry.referenceNode = &referenceNode;

  // Perform the evaluation between the roots of either tree.
  rootRefEntry.score = rule.Score(queryNode, referenceNode);
  rootRefEntry.baseCase = rule.BaseCase(queryNode.Point(),
      referenceNode.Point());
  rootRefEntry.traversalInfo = rule.TraversalInfo();

  refMap[referenceNode.Scale()].push_back(rootRefEntry);

  Traverse(queryNode, refMap);
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::Traverse(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry> >& referenceMap)
{
  if (referenceMap.size() == 0)
    return; // Nothing to do!

  // First recurse down the reference nodes as necessary.
  ReferenceRecursion(queryNode, referenceMap);

  // Did the map get emptied?
  if (referenceMap.size() == 0)
    return; // Nothing to do!

  // Now, reduce the scale of the query node by recursing.  But we can't recurse
  // if the query node is a leaf node.
  if ((queryNode.Scale() != INT_MIN) &&
      (queryNode.Scale() >= (*referenceMap.rbegin()).first))
  {
    // Recurse into the non-self-children first.  The recursion order cannot
    // affect the runtime of the algorithm, because each query child recursion's
    // results are separate and independent.  I don't think this is true in
    // every case, and we may have to modify this section to consider scores in
    // the future.
    for (size_t i = 1; i < queryNode.NumChildren(); ++i)
    {
      // We need a copy of the map for this child.
      std::map<int, std::vector<DualCoverTreeMapEntry> > childMap;
      PruneMap(queryNode.Child(i), referenceMap, childMap);
      Traverse(queryNode.Child(i), childMap);
    }
    std::map<int, std::vector<DualCoverTreeMapEntry> > selfChildMap;
    PruneMap(queryNode.Child(0), referenceMap, selfChildMap);
    Traverse(queryNode.Child(0), selfChildMap);
  }

  if (queryNode.Scale() != INT_MIN)
    return; // No need to evaluate base cases at this level.  It's all done.

  // If we have made it this far, all we have is a bunch of base case
  // evaluations to do.
  Log::Assert((*referenceMap.begin()).first == INT_MIN);
  Log::Assert(queryNode.Scale() == INT_MIN);
  std::vector<DualCoverTreeMapEntry>& pointVector =
      (*referenceMap.begin()).second;

  for (size_t i = 0; i < pointVector.size(); ++i)
  {
    // Get a reference to the frame.
    const DualCoverTreeMapEntry& frame = pointVector[i];

    CoverTree* refNode = frame.referenceNode;

    // If the point is the same as both parents, then we have already done this
    // base case.
    if ((refNode->Point() == refNode->Parent()->Point()) &&
        (queryNode.Point() == queryNode.Parent()->Point()))
    {
      ++numPrunes;
      continue;
    }

    // Score the node, to see if we can prune it, after restoring the traversal
    // info.
    rule.TraversalInfo() = frame.traversalInfo;
    double score = rule.Score(queryNode, *refNode);

    if (score == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }

    // If not, compute the base case.
    rule.BaseCase(queryNode.Point(), pointVector[i].referenceNode->Point());
  }
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::PruneMap(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry> >& referenceMap,
    std::map<int, std::vector<DualCoverTreeMapEntry> >& childMap)
{
  if (referenceMap.empty())
    return; // Nothing to do.

  // Copy the zero set first.
  if ((*referenceMap.begin()).first == INT_MIN)
  {
    // Get a reference to the vector representing the entries at this scale.
    std::vector<DualCoverTreeMapEntry>& scaleVector =
        (*referenceMap.begin()).second;

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    const int thisScale = (*referenceMap.begin()).first;
    childMap[thisScale].reserve(scaleVector.size());
    std::vector<DualCoverTreeMapEntry>& newScaleVector = childMap[thisScale];

    // Loop over each entry in the vector.
    for (size_t j = 0; j < scaleVector.size(); ++j)
    {
      const DualCoverTreeMapEntry& frame = scaleVector[j];

      // First evaluate if we can prune without performing the base case.
      CoverTree* refNode = frame.referenceNode;

      // Perform the actual scoring, after restoring the traversal info.
      rule.TraversalInfo() = frame.traversalInfo;
      double score = rule.Score(queryNode, *refNode);

      if (score == DBL_MAX)
      {
        // Pruned.  Move on.
        ++numPrunes;
        continue;
      }

      // If it isn't pruned, we must evaluate the base case.
      const double baseCase = rule.BaseCase(queryNode.Point(),
          refNode->Point());

      // Add to child map.
      newScaleVector.push_back(frame);
      newScaleVector.back().score = score;
      newScaleVector.back().baseCase = baseCase;
      newScaleVector.back().traversalInfo = rule.TraversalInfo();
    }

    // If we didn't add anything, then strike this vector from the map.
    if (newScaleVector.size() == 0)
      childMap.erase((*referenceMap.begin()).first);
  }

  typename std::map<int, std::vector<DualCoverTreeMapEntry> >::reverse_iterator
      it = referenceMap.rbegin();

  while ((it != referenceMap.rend()))
  {
    const int thisScale = (*it).first;
    if (thisScale == INT_MIN) // We already did it.
      break;

    // Get a reference to the vector representing the entries at this scale.
    std::vector<DualCoverTreeMapEntry>& scaleVector = (*it).second;

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    childMap[thisScale].reserve(scaleVector.size());
    std::vector<DualCoverTreeMapEntry>& newScaleVector = childMap[thisScale];

    // Loop over each entry in the vector.
    for (size_t j = 0; j < scaleVector.size(); ++j)
    {
      const DualCoverTreeMapEntry& frame = scaleVector[j];

      // First evaluate if we can prune without performing the base case.
      CoverTree* refNode = frame.referenceNode;

      // Perform the actual scoring, after restoring the traversal info.
      rule.TraversalInfo() = frame.traversalInfo;
      double score = rule.Score(queryNode, *refNode);

      if (score == DBL_MAX)
      {
        // Pruned.  Move on.
        ++numPrunes;
        continue;
      }

      // If it isn't pruned, we must evaluate the base case.
      const double baseCase = rule.BaseCase(queryNode.Point(),
          refNode->Point());

      // Add to child map.
      newScaleVector.push_back(frame);
      newScaleVector.back().score = score;
      newScaleVector.back().baseCase = baseCase;
      newScaleVector.back().traversalInfo = rule.TraversalInfo();
    }

    // If we didn't add anything, then strike this vector from the map.
    if (newScaleVector.size() == 0)
      childMap.erase((*it).first);

    ++it; // Advance to next scale.
  }
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::ReferenceRecursion(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry> >& referenceMap)
{
  // First, reduce the maximum scale in the reference map down to the scale of
  // the query node.
  while (!referenceMap.empty())
  {
    // Hacky bullshit to imitate jl cover tree.
    if (queryNode.Parent() == NULL && (*referenceMap.rbegin()).first <
        queryNode.Scale())
      break;
    if (queryNode.Parent() != NULL && (*referenceMap.rbegin()).first <=
        queryNode.Scale())
      break;
    // If the query node's scale is INT_MIN and the reference map's maximum
    // scale is INT_MIN, don't try to recurse...
    if ((queryNode.Scale() == INT_MIN) &&
       ((*referenceMap.rbegin()).first == INT_MIN))
      break;

    // Get a reference to the current largest scale.
    std::vector<DualCoverTreeMapEntry>& scaleVector = (*referenceMap.rbegin()).second;

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    // Now loop over each element.
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      // Get a reference to the current element.
      const DualCoverTreeMapEntry& frame = scaleVector.at(i);

      CoverTree* refNode = frame.referenceNode;

      // Create the score for the children.
      double score = rule.Rescore(queryNode, *refNode, frame.score);

      // Now if this childScore is DBL_MAX we can prune all children.  In this
      // recursion setup pruning is all or nothing for children.
      if (score == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      // If it is not pruned, we must evaluate the base case.

      // Add the children.
      for (size_t j = 0; j < refNode->NumChildren(); ++j)
      {
        rule.TraversalInfo() = frame.traversalInfo;
        double childScore = rule.Score(queryNode, refNode->Child(j));
        if (childScore == DBL_MAX)
        {
          ++numPrunes;
          continue;
        }

        // It wasn't pruned; evaluate the base case.
        const double baseCase = rule.BaseCase(queryNode.Point(),
            refNode->Child(j).Point());

        DualCoverTreeMapEntry newFrame;
        newFrame.referenceNode = &refNode->Child(j);
        newFrame.score = childScore; // Use the score of the parent.
        newFrame.baseCase = baseCase;
        newFrame.traversalInfo = rule.TraversalInfo();

        referenceMap[newFrame.referenceNode->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    referenceMap.erase((*referenceMap.rbegin()).first);
  }
}

} // namespace tree
} // namespace mlpack

#endif
