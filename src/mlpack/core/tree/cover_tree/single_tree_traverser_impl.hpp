/**
 * @file core/tree/cover_tree/single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the single tree traverser for cover trees, which implements
 * a breadth-first traversal.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

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
SingleTreeTraverser<RuleType>::SingleTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{
  hotScaleLevels.fill(INT_MIN);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
inline size_t
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
SingleTreeTraverser<RuleType>::ScaleIndex(const int scale)
{
  // If it's a leaf, we have to put it in the leaf vector.
  if (scale == INT_MIN)
    return size_t(-1);

  // Loop through the hot scales.  Allocate one, if it's unallocated.
  int minScale = INT_MAX;
  int firstMinScale = 8;
  for (size_t i = 0; i < 8; ++i)
  {
    if (hotScaleLevels[i] == INT_MIN)
      firstMinScale = i; // We will allocate this vector.
    if (hotScaleLevels[i] == scale)
      return i;
    if (hotScaleLevels[i] < minScale)
      minScale = hotScaleLevels[i];
  }
  if (firstMinScale != 8)
  {
    hotScaleLevels[firstMinScale] = scale;
    return firstMinScale;
  }

  if (scale > minScale)
  {
    // In this case, we are encountering a scale that *should* be in the hot
    // scale set but isn't!  So, we need to evict an existing one.
    size_t minScaleIndex = 8;
    minScale = INT_MAX;
    for (size_t i = 0; i < 8; ++i)
    {
      if (hotScaleLevels[i] < minScale)
      {
        minScaleIndex = i;
        minScale = hotScaleLevels[i];
      }
    }

    // Move the smallest-scale vector into the cold map and claim the hot
    // vector for this scale.
    mapQueue[minScale] = std::move(hotScaleVectors[minScaleIndex]);
    hotScaleLevels[minScaleIndex] = scale;
    return minScaleIndex;
  }

  // If there are no hot scales for this scale level, we will need to put it in
  // the std::map.
  return 8;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
SingleTreeTraverser<RuleType>::Traverse(
    const size_t queryIndex,
    CoverTree& referenceNode)
{
  // This is a non-recursive implementation (which should be faster than a
  // recursive implementation).

  // Generally, a single-tree traversal will only have a couple active scale
  // levels at any given time; the vast majority of times we descend any
  // reference node, the child we descend to is either the next scale level, or
  // perhaps two scale levels away.  Therefore, we maintain a "hot" set of scale
  // vectors for quick lookup, and in the odd case that we get something at a
  // totally different level, we use a std::map as a priority queue.  These are
  // all members of the SingleTreeTraverser class.

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
      CoverTreeMapEntry newFrame;
      newFrame.node = &referenceNode.Child(i);
      newFrame.score = rootChildScore;
      newFrame.baseCase = rootBaseCase;

      // Put it into the map.
      const size_t scaleIndex = ScaleIndex(newFrame.node->Scale());
      if (scaleIndex < 8)
        hotScaleVectors[scaleIndex].push_back(std::move(newFrame));
      else if (scaleIndex == size_t(-1))
        leafVector.push_back(std::move(newFrame));
      else
        mapQueue[newFrame.node->Scale()].push_back(newFrame);
    }
  }

  // Now begin the iteration through the levels of the tree.  We will treat the
  // leaves differently (below).
  size_t maxIndex = hotScaleLevels.index_max();
  while (hotScaleLevels[maxIndex] != INT_MIN)
  {
    // Get a reference to the current scale.
    std::vector<CoverTreeMapEntry>& scaleVector = hotScaleVectors[maxIndex];

    // Iterate over all the nodes at the current scale, pruning if we can.
    size_t firstGoodIndex = 0;
    for (size_t i = 0; i < scaleVector.size(); ++i)
    {
      CoverTreeMapEntry& frame = scaleVector.at(i);
      if (rule.Rescore(queryIndex, *frame.node, frame.score) == DBL_MAX)
      {
        ++numPrunes;
        frame.score = DBL_MAX;
        if (i > firstGoodIndex)
        {
          std::swap(frame, scaleVector.at(firstGoodIndex));
          ++firstGoodIndex;
        }
        continue;
      }

      frame.score = rule.Score(queryIndex, *frame.node);
      if (frame.score == DBL_MAX)
      {
        ++numPrunes;
        if (i > firstGoodIndex)
        {
          std::swap(frame, scaleVector.at(firstGoodIndex));
          ++firstGoodIndex;
        }
        continue;
      }

      // Compute the base case if we are not a self-child.
      const size_t point = frame.node->Point();
      if (frame.node->Parent()->Point() != point)
        frame.baseCase = rule.BaseCase(queryIndex, point);
    }

    // For what remains, sort the vector so that we compute base cases with the
    // most likely good nodes first.
    std::sort(scaleVector.begin() + firstGoodIndex, scaleVector.end());

    // In a second pass, add the children of nodes that weren't pruned.
    for (size_t i = firstGoodIndex; i < scaleVector.size(); ++i)
    {
      const CoverTreeMapEntry& frame = scaleVector.at(i);
      // One more pruning attempt.
      if (frame.score == DBL_MAX ||
          rule.Rescore(queryIndex, *frame.node, frame.score) == DBL_MAX)
      {
        numPrunes += frame.node->NumChildren();
        continue;
      }

      for (size_t c = 0; c < frame.node->NumChildren(); ++c)
      {
        // Don't add a self-leaf.
        if (c == 0 && frame.node->Child(0).NumChildren() == 0)
        {
          ++numPrunes;
          continue;
        }

        CoverTreeMapEntry newFrame;
        newFrame.node = &frame.node->Child(c);
        newFrame.score = frame.score;
        newFrame.baseCase = frame.baseCase;

        // Determine which vector to add the child to.
        const size_t scaleIndex = ScaleIndex(newFrame.node->Scale());
        if (scaleIndex < 8)
          hotScaleVectors[scaleIndex].push_back(std::move(newFrame));
        else if (scaleIndex == size_t(-1))
          leafVector.push_back(std::move(newFrame));
        else
          mapQueue[newFrame.node->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    if (!mapQueue.empty())
    {
      // Reallocate the hot vector to the next largest scale level we have been
      // collecting points for.
      hotScaleVectors[maxIndex] = std::move(mapQueue.begin()->second);
      hotScaleLevels[maxIndex] = mapQueue.begin()->first;
      mapQueue.erase(mapQueue.begin()->first);
    }
    else
    {
      hotScaleVectors[maxIndex].clear();
      hotScaleLevels[maxIndex] = INT_MIN;
    }

    maxIndex = hotScaleLevels.index_max();
  }

  for (size_t i = 0; i < leafVector.size(); ++i)
  {
    CoverTreeMapEntry& frame = leafVector.at(i);

    CoverTree* node = frame.node;
    const double score = frame.score;
    const size_t point = node->Point();

    // First, recalculate the score of this node to find if we can prune it.
    if (rule.Rescore(queryIndex, *node, score) == DBL_MAX)
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

  leafVector.clear(); // For next call to Traverse().
}

} // namespace mlpack

#endif
