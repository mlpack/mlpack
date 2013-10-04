/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 *
 * This file is part of MLPACK 1.0.7.
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
#ifndef __MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

#include <mlpack/core.hpp>
#include <queue>

namespace mlpack {
namespace tree {

//! The object placed in the map for tree traversal.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
struct DualCoverTreeMapEntry
{
  //! The node this entry refers to.
  CoverTree<MetricType, RootPointPolicy, StatisticType>* referenceNode;
  //! The score of the node.
  double score;

  //! Comparison operator, for sorting within the map.
  bool operator<(const DualCoverTreeMapEntry& other) const
  {
    return (score < other.score);
  }
};

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::Traverse(
    CoverTree<MetricType, RootPointPolicy, StatisticType>& queryNode,
    CoverTree<MetricType, RootPointPolicy, StatisticType>& referenceNode)
{
  typedef DualCoverTreeMapEntry<MetricType, RootPointPolicy, StatisticType>
      MapEntryType;

  // Start by creating a map and adding the reference root node to it.
  std::map<int, std::vector<MapEntryType> > refMap;

  MapEntryType rootRefEntry;

  rootRefEntry.referenceNode = &referenceNode;
  rootRefEntry.score = 0.0; // Must recurse into.

  refMap[referenceNode.Scale()].push_back(rootRefEntry);

  Traverse(queryNode, refMap);
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::Traverse(
    CoverTree<MetricType, RootPointPolicy, StatisticType>& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry<MetricType, RootPointPolicy,
        StatisticType> > >& referenceMap)
{
  // Convenience typedef.
  typedef DualCoverTreeMapEntry<MetricType, RootPointPolicy, StatisticType>
      MapEntryType;

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
    std::map<int, std::vector<MapEntryType> > childMap;
    PruneMap(queryNode, referenceMap, childMap);
    for (size_t i = 0; i < queryNode.NumChildren(); ++i)
    {
      // We need a copy of the map for this child.
      std::map<int, std::vector<MapEntryType> > thisChildMap = childMap;
      Traverse(queryNode.Child(i), thisChildMap);
    }
  }

  if (queryNode.Scale() != INT_MIN)
    return; // No need to evaluate base cases at this level.  It's all done.

  // If we have made it this far, all we have is a bunch of base case
  // evaluations to do.
  Log::Assert((*referenceMap.begin()).first == INT_MIN);
  Log::Assert(queryNode.Scale() == INT_MIN);
  std::vector<MapEntryType>& pointVector = (*referenceMap.begin()).second;

  for (size_t i = 0; i < pointVector.size(); ++i)
  {
    // Get a reference to the frame.
    const MapEntryType& frame = pointVector[i];

    CoverTree<MetricType, RootPointPolicy, StatisticType>* refNode =
        frame.referenceNode;

    // If the point is the same as both parents, then we have already done this
    // base case.
    if ((refNode->Point() == refNode->Parent()->Point()) &&
        (queryNode.Point() == queryNode.Parent()->Point()))
    {
      ++numPrunes;
      continue;
    }

    // Score the node, to see if we can prune it.
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

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::PruneMap(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry<MetricType,
        RootPointPolicy, StatisticType> > >& referenceMap,
    std::map<int, std::vector<DualCoverTreeMapEntry<MetricType,
        RootPointPolicy, StatisticType> > >& childMap)
{
  typedef DualCoverTreeMapEntry<MetricType, RootPointPolicy, StatisticType>
      MapEntryType;

  if (referenceMap.empty())
    return; // Nothing to do.
  typename std::map<int, std::vector<MapEntryType> >::reverse_iterator it =
      referenceMap.rbegin();

  while ((it != referenceMap.rend()))
  {
    // Get a reference to the vector representing the entries at this scale.
    std::vector<MapEntryType>& scaleVector = (*it).second;

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    const int thisScale = (*it).first;
    childMap[thisScale].reserve(scaleVector.size());
    std::vector<MapEntryType>& newScaleVector = childMap[thisScale];

    // Loop over each entry in the vector.
    for (size_t j = 0; j < scaleVector.size(); ++j)
    {
      const MapEntryType& frame = scaleVector[j];

      // First evaluate if we can prune without performing the base case.
      CoverTree<MetricType, RootPointPolicy, StatisticType>* refNode =
          frame.referenceNode;

      // Perform the actual scoring.
      double score = rule.Score(queryNode, *refNode);

      if (score == DBL_MAX)
      {
        // Pruned.  Move on.
        ++numPrunes;
        continue;
      }

      // If it isn't pruned, we must evaluate the base case.
      rule.BaseCase(queryNode.Point(), refNode->Point());

      // Add to child map.
      newScaleVector.push_back(frame);
      newScaleVector.back().score = score;
    }

    // If we didn't add anything, then strike this vector from the map.
    if (newScaleVector.size() == 0)
      childMap.erase((*it).first);

    ++it; // Advance to next scale.
  }
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::ReferenceRecursion(
    CoverTree& queryNode,
    std::map<int, std::vector<DualCoverTreeMapEntry<MetricType, RootPointPolicy,
        StatisticType> > >& referenceMap)
{
  typedef DualCoverTreeMapEntry<MetricType, RootPointPolicy, StatisticType>
      MapEntryType;

  // First, reduce the maximum scale in the reference map down to the scale of
  // the query node.
  while (!referenceMap.empty() &&
      ((*referenceMap.rbegin()).first > queryNode.Scale()))
  {
    // If the query node's scale is INT_MIN and the reference map's maximum
    // scale is INT_MIN, don't try to recurse...
    if ((queryNode.Scale() == INT_MIN) &&
       ((*referenceMap.rbegin()).first == INT_MIN))
      break;

    // Get a reference to the current largest scale.
    std::vector<MapEntryType>& scaleVector = (*referenceMap.rbegin()).second;

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    // Now loop over each element.
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      // Get a reference to the current element.
      const MapEntryType& frame = scaleVector.at(i);

      CoverTree<MetricType, RootPointPolicy, StatisticType>* refNode =
          frame.referenceNode;

      // Create the score for the children.
      double score = rule.Score(queryNode, *refNode);

      // Now if this childScore is DBL_MAX we can prune all children.  In this
      // recursion setup pruning is all or nothing for children.
      if (score == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      // If it is not pruned, we must evaluate the base case.
      rule.BaseCase(queryNode.Point(), refNode->Point());

      // Add the children.
      for (size_t j = 0; j < refNode->NumChildren(); ++j)
      {
//        const size_t queryIndex = queryNode.Point();
//        const size_t refIndex = refNode->Child(j).Point();

        MapEntryType newFrame;
        newFrame.referenceNode = &refNode->Child(j);
        newFrame.score = score; // Use the score of the parent.

        referenceMap[newFrame.referenceNode->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    referenceMap.erase((*referenceMap.rbegin()).first);
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
