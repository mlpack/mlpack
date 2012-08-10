/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
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
  //! The index of the parent reference node.
  size_t referenceParent;
  //! The base case evaluation.
  double baseCase;
  //! The query node used for the base case evaluation (and/or score).
  CoverTree<MetricType, RootPointPolicy, StatisticType>* parentQueryNode;

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

  // Start by creating a map and adding the reference node to it.
  std::map<int, std::vector<MapEntryType> > refMap;

  MapEntryType rootRefEntry;

  rootRefEntry.referenceNode = &referenceNode;
  rootRefEntry.score = 0.0; // Must recurse into.
  rootRefEntry.referenceParent = (size_t() - 1); // Invalid index.
  rootRefEntry.baseCase = 0.0; // Not evaluated.
  rootRefEntry.parentQueryNode = &queryNode; // No query node was used yet.

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
//  Log::Debug << "Recursed into query node " << queryNode.Point() << ", scale "
//      << queryNode.Scale() << "\n";

  // Convenience typedef.
  typedef DualCoverTreeMapEntry<MetricType, RootPointPolicy, StatisticType>
      MapEntryType;

  // First, reduce the maximum scale in the reference map down to the scale of
  // the query node.
  while ((*referenceMap.rbegin()).first > queryNode.Scale())
  {
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
      CoverTree<MetricType, RootPointPolicy, StatisticType>* parentQueryNode =
          frame.parentQueryNode;
      const double score = frame.score;
      const size_t refParent = frame.referenceParent;
      const size_t refPoint = refNode->Point();
      const size_t parentQueryPoint = parentQueryNode->Point();
      double baseCase = frame.baseCase;

//      Log::Debug << "Currently inspecting reference node " << refNode->Point()
//          << " scale " << refNode->Scale() << " parentQueryPoint " <<
//          parentQueryPoint << std::endl;

//      Log::Debug << "Old score " << score << " with refParent " << refParent
//          << " and queryParent " << parentQueryNode->Point() << " scale " <<
//          parentQueryNode->Scale() << "\n";

      // First we recalculate the score of this node to find if we can prune it.
      if (rule.Rescore(queryNode, *refNode, score) == DBL_MAX)
      {
//        Log::Warn << "Pruned after rescore\n";
        ++numPrunes;
        continue;
      }

      // If this is a self-child, the base case has already been evaluated.
      // We also must ensure that the base case was evaluated with this query
      // point.
      if ((refPoint != refParent) || (queryNode.Point() != parentQueryPoint))
      {
//        Log::Warn << "Must evaluate base case\n";
        baseCase = rule.BaseCase(queryNode.Point(), refPoint);
//        Log::Debug << "Base case " << baseCase << std::endl;
        rule.UpdateAfterRecursion(queryNode, *refNode); // Kludgey.
//       Log::Debug << "Bound for point " << queryNode.Point() << " scale " <<
//            queryNode.Scale() << " is now " << queryNode.Stat().Bound() <<
//            std::endl;
      }

      // Create the score for the children.
      const double childScore = rule.Score(queryNode, *refNode, baseCase);

      // Now if this childScore is DBL_MAX we can prune all children.  In this
      // recursion setup pruning is all or nothing for children.
      if (childScore == DBL_MAX)
      {
//        Log::Warn << "Pruned all children.\n";
        numPrunes += refNode->NumChildren();
        continue;
      }

      // In the dual recursion we must add the self-leaf (as compared to the
      // single recursion); in this case we have potentially more than one point
      // under the query node, so we cannot prune the self-leaf.
      for (size_t j = 0; j < refNode->NumChildren(); ++j)
      {
        MapEntryType newFrame;
        newFrame.referenceNode = &refNode->Child(j);
        newFrame.score = childScore;
        newFrame.baseCase = baseCase;
        newFrame.referenceParent = refPoint;
        newFrame.parentQueryNode = &queryNode;

//        Log::Debug << "Push onto map child " << refNode->Child(j).Point() <<
//            " scale " << refNode->Child(j).Scale() << std::endl;

        referenceMap[newFrame.referenceNode->Scale()].push_back(newFrame);
      }
    }

    // Now clear the memory for this scale; it isn't needed anymore.
    referenceMap.erase((*referenceMap.rbegin()).first);
  }

  // Now, reduce the scale of the query node by recursing.  But we can't recurse
  // if the query node is a leaf node.
  if ((queryNode.Scale() != INT_MIN) &&
      (queryNode.Scale() >= (*referenceMap.rbegin()).first))
  {
    // Recurse into the non-self-children first.
    for (size_t i = 1; i < queryNode.NumChildren(); ++i)
    {
      std::map<int, std::vector<MapEntryType> > childMap(referenceMap);
//      Log::Debug << "Recurse into query child " << i << ": " <<
//          queryNode.Child(i).Point() << " scale " << queryNode.Child(i).Scale()
//          << "; this parent is " << queryNode.Point() << " scale " <<
//          queryNode.Scale() << std::endl;
      Traverse(queryNode.Child(i), childMap);
    }

    // Now we can use the existing map (without a copy) for the self-child.
//    Log::Debug << "Recurse into query self-child " << queryNode.Child(0).Point()
//        << " scale " << queryNode.Child(0).Scale() << "; this parent is "
//        << queryNode.Point() << " scale " << queryNode.Scale() << std::endl;
    Traverse(queryNode.Child(0), referenceMap);
  }

  if (queryNode.Scale() != INT_MIN)
    return; // No need to evaluate base cases at this level.  It's all done.

  // If we have made it this far, all we have is a bunch of base case
  // evaluations to do.
  Log::Assert((*referenceMap.begin()).first == INT_MIN);
  Log::Assert(queryNode.Scale() == INT_MIN);
  std::vector<MapEntryType>& pointVector = (*referenceMap.begin()).second;
//  Log::Debug << "Onto base case evaluations\n";
  for (size_t i = 0; i < pointVector.size(); ++i)
  {
    // Get a reference to the frame.
    const MapEntryType& frame = pointVector[i];

    CoverTree<MetricType, RootPointPolicy, StatisticType>* refNode =
        frame.referenceNode;
    CoverTree<MetricType, RootPointPolicy, StatisticType>* parentQueryNode =
        frame.parentQueryNode;
    const double oldScore = frame.score;
    const size_t refParent = frame.referenceParent;
//    Log::Debug << "Consider query " << queryNode.Point() << ", reference "
//        << refNode->Point() << "\n";
//    Log::Debug << "Old score " << oldScore << " with refParent " << refParent
//        << " and parent query node " << parentQueryNode->Point() << " scale "
//        << parentQueryNode->Scale() << std::endl;

    // First, ensure that we have not already calculated the base case.
    if ((parentQueryNode->Point() == queryNode.Point()) &&
        (refParent == refNode->Point()))
    {
//      Log::Debug << "Pruned because we already did the base case.\n";
      ++numPrunes;
      continue;
    }

    // Now, check if we can prune it.
    const double rescore = rule.Rescore(queryNode, *refNode, oldScore);

    if (rescore == DBL_MAX)
    {
//      Log::Debug << "Pruned after rescoring\n";
      ++numPrunes;
      continue;
    }

    // If not, compute the base case.
//    Log::Debug << "Not pruned, performing base case\n";
    rule.BaseCase(queryNode.Point(), pointVector[i].referenceNode->Point());
    rule.UpdateAfterRecursion(queryNode, *pointVector[i].referenceNode);
//    Log::Debug << "Bound for point " << queryNode.Point() << " scale " <<
//        queryNode.Scale() << " is now " << queryNode.Stat().Bound() <<
//        std::endl;
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
