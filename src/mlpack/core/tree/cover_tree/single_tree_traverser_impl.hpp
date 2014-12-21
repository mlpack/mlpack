/**
 * @file single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the single tree traverser for cover trees, which implements
 * a breadth-first traversal.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

#include <queue>

namespace mlpack {
namespace tree {

//! This is the structure the cover tree map will use for traversal.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
struct CoverTreeMapEntry
{
  //! The node this entry refers to.
  CoverTree<MetricType, RootPointPolicy, StatisticType>* node;
  //! The score of the node.
  double score;
  //! The index of the parent node.
  size_t parent;
  //! The base case evaluation.
  double baseCase;

  //! Comparison operator.
  bool operator<(const CoverTreeMapEntry& other) const
  {
    return (score < other.score);
  }
};

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
CoverTree<MetricType, RootPointPolicy, StatisticType>::
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    CoverTree<MetricType, RootPointPolicy, StatisticType>& referenceNode)
{
  // This is a non-recursive implementation (which should be faster than a
  // recursive implementation).
  typedef CoverTreeMapEntry<MetricType, RootPointPolicy, StatisticType>
      MapEntryType;

  // We will use this map as a priority queue.  Each key represents the scale,
  // and then the vector is all the nodes in that scale which need to be
  // investigated.  Because no point in a scale can add a point in its own
  // scale, we know that the vector for each scale is final when we get to it.
  // In addition, map is organized in such a way that rbegin() will return the
  // largest scale.
  std::map<int, std::vector<MapEntryType> > mapQueue;

  // Create the score for the children.
  double rootChildScore = rule.Score(queryIndex, referenceNode);

  if (rootChildScore == DBL_MAX)
  {
    numPrunes += referenceNode.NumChildren();
  }
  else
  {
    // Manually add the children of the first node.
    // Often, a ruleset will return without doing any computation on cover trees
    // using TreeTraits::FirstPointIsCentroid; this is an optimization that
    // (theoretically) the compiler should get right.
    double rootBaseCase = rule.BaseCase(queryIndex, referenceNode.Point());

    // Don't add the self-leaf.
    size_t i = 0;
    if (referenceNode.Child(0).NumChildren() == 0)
    {
      ++numPrunes;
      i = 1;
    }

    for (/* i was set above. */; i < referenceNode.NumChildren(); ++i)
    {
      MapEntryType newFrame;
      newFrame.node = &referenceNode.Child(i);
      newFrame.score = rootChildScore;
      newFrame.baseCase = rootBaseCase;
      newFrame.parent = referenceNode.Point();

      // Put it into the map.
      mapQueue[newFrame.node->Scale()].push_back(newFrame);
    }
  }

  // Now begin the iteration through the map, but only if it has anything in it.
  if (mapQueue.empty())
    return;
  typename std::map<int, std::vector<MapEntryType> >::reverse_iterator rit =
      mapQueue.rbegin();

  // We will treat the leaves differently (below).
  while ((*rit).first != INT_MIN)
  {
    // Get a reference to the current scale.
    std::vector<MapEntryType>& scaleVector = (*rit).second;

    // Before traversing all the points in this scale, sort by score.
    std::sort(scaleVector.begin(), scaleVector.end());

    // Now loop over each element.
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      // Get a reference to the current element.
      const MapEntryType& frame = scaleVector.at(i);

      CoverTree<MetricType, RootPointPolicy, StatisticType>* node = frame.node;
      const double score = frame.score;
      const size_t parent = frame.parent;
      const size_t point = node->Point();
      double baseCase = frame.baseCase;

      // First we recalculate the score of this node to find if we can prune it.
      if (rule.Rescore(queryIndex, *node, score) == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      // Create the score for the children.
      const double childScore = rule.Score(queryIndex, *node);

      // Now if this childScore is DBL_MAX we can prune all children.  In this
      // recursion setup pruning is all or nothing for children.
      if (childScore == DBL_MAX)
      {
        numPrunes += node->NumChildren();
        continue;
      }

      // If we are a self-child, the base case has already been evaluated.
      // Often, a ruleset will return without doing any computation on cover
      // trees using TreeTraits::FirstPointIsCentroid; this is an optimization
      // that (theoretically) the compiler should get right.
      if (point != parent)
        baseCase = rule.BaseCase(queryIndex, point);

      // Don't add the self-leaf.
      size_t j = 0;
      if (node->Child(0).NumChildren() == 0)
      {
        ++numPrunes;
        j = 1;
      }

      for (/* j is already set. */; j < node->NumChildren(); ++j)
      {
        MapEntryType newFrame;
        newFrame.node = &node->Child(j);
        newFrame.score = childScore;
        newFrame.baseCase = baseCase;
        newFrame.parent = point;

        mapQueue[newFrame.node->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    mapQueue.erase((*rit).first);
  }

  // Now deal with the leaves.
  for (size_t i = 0; i < mapQueue[INT_MIN].size(); ++i)
  {
    const MapEntryType& frame = mapQueue[INT_MIN].at(i);

    CoverTree<MetricType, RootPointPolicy, StatisticType>* node = frame.node;
    const double score = frame.score;
    const size_t point = node->Point();

    // First, recalculate the score of this node to find if we can prune it.
    double rescore = rule.Rescore(queryIndex, *node, score);

    if (rescore == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }

    // For this to be a valid dual-tree algorithm, we *must* evaluate the
    // combination, even if pruning it will make no difference.  It's the
    // definition.
    const double actualScore = rule.Score(queryIndex, *node);

    if (actualScore == DBL_MAX)
    {
      ++numPrunes;
      continue;
    }
    else
    {
      // Evaluate the base case, since the combination was not pruned.
      // Often, a ruleset will return without doing any computation on cover
      // trees using TreeTraits::FirstPointIsCentroid; this is an optimization
      // that (theoretically) the compiler should get right.
      rule.BaseCase(queryIndex, point);
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
