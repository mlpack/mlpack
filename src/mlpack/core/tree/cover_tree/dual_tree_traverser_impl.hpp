/**
 * @file core/tree/cover_tree/dual_tree_traverser_impl.hpp
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

#include <mlpack/core/util/log.hpp>
#include <mlpack/prereqs.hpp>
#include <queue>

namespace mlpack {

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::Traverse(CoverTree& queryRoot,
                                      CoverTree& referenceNode)
{
  // Start by creating the recursion sets and adding the reference node to it.
  typedef CoverTreeRecursionSets<MapEntryType, 8> RecursionSets;
  RecursionSets rootRecursionSets;

  // Perform the evaluation between the roots of either tree.
  const double rootScore = rule.Score(queryRoot, referenceNode);
  const double rootBaseCase = rule.BaseCase(queryRoot.Point(),
      referenceNode.Point());
  MapEntryType rootRefEntry(&referenceNode, rootScore, rootBaseCase,
      rule.TraversalInfo());

  rootRecursionSets.GetScaleVector(referenceNode.Scale()).push_back(
      rootRefEntry);

  std::stack<std::pair<CoverTree*, RecursionSets>> queryStack;
  queryStack.push(std::make_pair(&queryRoot, std::move(rootRecursionSets)));
  while (!queryStack.empty())
  {
    CoverTree* queryNode = queryStack.top().first;
    RecursionSets recursionSets = std::move(queryStack.top().second);

    queryStack.pop();

    if (recursionSets.IsEmpty())
      continue;

    // First recurse down the reference nodes as necessary.
    ReferenceRecursion(*queryNode, recursionSets);

    // Did the map get emptied?
    if (recursionSets.IsEmpty())
      continue; // Nothing to do!

    // Now, reduce the scale of the query node by recursing.  But we can't
    // recurse if the query node is a leaf node.
    if ((queryNode->Scale() != INT_MIN) &&
        (queryNode->Scale() >= recursionSets.MaxScale()))
    {
      // Recurse into the non-self-children first.  The recursion order cannot
      // affect the runtime of the algorithm, because each query child
      // recursion's results are separate and independent.  I don't think this
      // is true in every case, and we may have to modify this section to
      // consider scores in the future.
      for (size_t i = 0; i < queryNode->NumChildren(); ++i)
      {
        // We need a copy of the map for this child.
        RecursionSets childRecursionSets;
        PruneMap(queryNode->Child(i), recursionSets, childRecursionSets);
        queryStack.push(std::make_pair(&queryNode->Child(i),
            std::move(childRecursionSets)));
      }
      continue;
    }

    if (queryNode->Scale() != INT_MIN)
      continue; // No need to evaluate base cases at this level.  It's all done.

    // If we have made it this far, all we have is a bunch of base case
    // evaluations to do.
    Log::Assert(recursionSets.MaxScale() == INT_MIN);
    Log::Assert(queryNode->Scale() == INT_MIN);
    std::vector<MapEntryType>& pointVector =
        recursionSets.GetScaleVector(INT_MIN);

    for (size_t i = 0; i < pointVector.size(); ++i)
    {
      // Get a reference to the frame.
      const MapEntryType& frame = pointVector[i];

      CoverTree* refNode = frame.node;

      // If the point is the same as both parents, then we have already done
      // this base case.
      if ((refNode->Point() == refNode->Parent()->Point()) &&
          (queryNode->Point() == queryNode->Parent()->Point()))
      {
        ++numPrunes;
        continue;
      }

      // Score the node, to see if we can prune it, after restoring the
      // traversal info.
      rule.TraversalInfo() = frame.traversalInfo;
      double score = rule.Score(*queryNode, *refNode);

      if (score == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      // If not, compute the base case.
      rule.BaseCase(queryNode->Point(), pointVector[i].node->Point());
    }
  }
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::PruneMap(
    CoverTree& queryNode,
    CoverTreeRecursionSets<MapEntryType, 8>& recursionSets,
    CoverTreeRecursionSets<MapEntryType, 8>& childRecursionSets)
{
  if (recursionSets.IsEmpty())
    return; // Nothing to do.

  typename CoverTreeRecursionSets<MapEntryType, 8>::Iterator it =
      recursionSets.begin();
  while (!it.AtEnd())
  {
    // Get a reference to the vector representing the entries at this scale.
    std::vector<MapEntryType>& scaleVector = it.Vector();
    const int scale = it.Scale();
    std::vector<MapEntryType>& childScaleVector =
        childRecursionSets.GetScaleVector(scale);

    // Before traversing all the points in this scale, sort by score.  Don't
    // bother sorting the leaf set, because getting that in the right order
    // isn't likely to affect traversal time in any meaningful way.
    if (scale != INT_MIN)
      std::sort(scaleVector.begin(), scaleVector.end());

    childScaleVector.reserve(scaleVector.size());

    // Loop over each entry in the vector.
    for (size_t j = 0; j < scaleVector.size(); ++j)
    {
      const MapEntryType& frame = scaleVector[j];

      // First evaluate if we can prune without performing the base case.
      CoverTree* refNode = frame.node;

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
      childScaleVector.emplace_back(frame.node, score, baseCase,
          rule.TraversalInfo());
    }

    // If we didn't add anything, then strike this vector from the map.
    if (childScaleVector.size() == 0)
      childRecursionSets.RemoveScale(scale);

    ++it; // Advance to next scale.
  }
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
DualTreeTraverser<RuleType>::ReferenceRecursion(
    CoverTree& queryNode,
    CoverTreeRecursionSets<MapEntryType, 8>& recursionSets)
{
  // First, reduce the maximum scale in the reference map down to the scale of
  // the query node.
  while (!recursionSets.IsEmpty())
  {
    // Start from the largest scale and keep recursing until we are done.
    const int maxScale = recursionSets.MaxScale();
    // Hacky bullshit to imitate jl cover tree.
    if (queryNode.Parent() == NULL && maxScale < queryNode.Scale())
      break;
    if (queryNode.Parent() != NULL && maxScale <= queryNode.Scale())
      break;
    // If the query node's scale is INT_MIN and the reference map's maximum
    // scale is INT_MIN, don't try to recurse...
    if (queryNode.Scale() == INT_MIN && maxScale == INT_MIN)
      break;

    // Get a reference to the current largest scale.
    std::vector<MapEntryType>& scaleVector =
        recursionSets.GetScaleVector(maxScale);

    // Now loop over each element.
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      // Get a reference to the current element.
      const MapEntryType& frame = scaleVector.at(i);
      CoverTree* refNode = frame.node;

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

        const int newScale = refNode->Child(j).Scale();
        recursionSets.GetScaleVector(newScale).emplace_back(&refNode->Child(j),
            childScore, // Use the score of the parent.
            baseCase,
            rule.TraversalInfo());
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    recursionSets.RemoveScale(maxScale);
  }
}

} // namespace mlpack

#endif
