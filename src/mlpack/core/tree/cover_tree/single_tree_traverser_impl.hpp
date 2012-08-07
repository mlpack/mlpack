/**
 * @file single_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the single tree traverser for cover trees, which implements
 * a breadth-first traversal.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "single_tree_traverser.hpp"

#include <queue>

namespace mlpack {
namespace tree {

//! This is the structure the priority queue will use for traversal.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
struct CoverTreeQueueEntry
{
  //! The node this entry refers to.
  CoverTree<MetricType, RootPointPolicy, StatisticType>* node;
  //! The score of the node.
  double score;
  //! The index of the parent node.
  size_t parent;
  //! The base case evaluation (-1.0 if it has not been performed).
  double baseCase;

  //! Comparison operator.
  bool operator<(const CoverTreeQueueEntry& other) const
  {
    return ((node->Scale() < other.node->Scale()) ||
            ((node->Scale() == other.node->Scale()) && (score < other.score)));
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
  typedef CoverTreeQueueEntry<MetricType, RootPointPolicy, StatisticType>
      QueueType;
  std::priority_queue<QueueType> pointQueue;

  // Unsorted list of leaves we have to look through.
  std::queue<QueueType> leafQueue;

  QueueType first;
  first.node = &referenceNode;
  first.score = 0.0;
  first.parent = (size_t() - 1); // Invalid index.
  first.baseCase = -1.0;
  pointQueue.push(first);

  Log::Warn << "Beginning recursion for " << queryIndex << std::endl;

  while (!pointQueue.empty())
  {
    QueueType frame = pointQueue.top();

    CoverTree<MetricType, RootPointPolicy, StatisticType>* node = frame.node;
    const double score = frame.score;
    const size_t parent = frame.parent;
    const size_t point = node->Point();
    double baseCase = frame.baseCase;

    Log::Debug << "Current point is " << node->Point() << " and scale "
        << node->Scale() << ".\n";

    pointQueue.pop();

    // First we (re)calculate the score of this node to find if we can prune it.
    Log::Debug << "Before rescoring, score is " << score << " and base case of "
        << "parent is " << baseCase << std::endl;
    double actualScore = rule.Rescore(queryIndex, *node, score);

//    Log::Debug << "Actual score is " << actualScore << ".\n";

    if (actualScore == DBL_MAX)
    {
      // Prune this node.
      Log::Debug << "Pruning after re-scoring (original score " << score << ")."
          << std::endl;
      ++numPrunes;
      continue; // Skip to next in queue.
    }

    // If we are a self-child, the base case has already been evaluated.
    if (point != parent)
    {
      baseCase = rule.BaseCase(queryIndex, point);
      Log::Debug << "Base case between " << queryIndex << " and " << point <<
          " evaluates to " << baseCase << ".\n";
    }
    else
    {
      Log::Debug << "Base case between " << queryIndex << " and " << point <<
          " already known to be " << baseCase << ".\n";
    }

    // Create the score for the children.
    double childScore = rule.Score(queryIndex, *node, baseCase);

    // Now if the childScore is DBL_MAX we can prune all children.  In this
    // recursion setup pruning is all or nothing for children.
    if (childScore == DBL_MAX)
    {
      Log::Debug << "Pruning all children.\n";
      numPrunes += node->NumChildren();
    }
    else
    {
      for (size_t i = 0; i < node->NumChildren(); ++i)
      {
        QueueType newFrame;
        newFrame.node = &node->Child(i);
        newFrame.score = childScore;
        newFrame.baseCase = baseCase;
        newFrame.parent = point;

        // Put it into the regular priority queue if it has children.
        if (newFrame.node->NumChildren() > 0)
        {
          Log::Debug << "Push back child " << i << ": point " <<
              newFrame.node->Point() << ", scale " << newFrame.node->Scale()
              << ".\n";
          pointQueue.push(newFrame);
        }
        else if ((newFrame.node->NumChildren() == 0) && (i > 0))
        {
          // We don't add the self-leaf to the leaf queue (it can't possibly
          // help).
          Log::Debug << "Push back child " << i << ": point " <<
              newFrame.node->Point() << ", scale " << newFrame.node->Scale()
              << ".\n";
          leafQueue.push(newFrame);
        }
        else
        {
          Log::Debug << "Prune self-leaf point " << point << ".\n";
          ++numPrunes;
        }
      }
    }
  }

  // Now look through all the leaves.
  while (!leafQueue.empty())
  {
    QueueType frame = leafQueue.front();

    CoverTree<MetricType, RootPointPolicy, StatisticType>* node = frame.node;
    const double score = frame.score;
    const size_t point = node->Point();

    Log::Debug << "Inspecting leaf " << point << " with score " << score <<
      "\n";

    leafQueue.pop();

    // First, recalculate the score of this node to find if we can prune it.
    double actualScore = rule.Rescore(queryIndex, *node, score);

    if (actualScore == DBL_MAX)
    {
      Log::Debug << "Pruned before base case.\n";
      ++numPrunes;
      continue;
    }

    // There are no self-leaves in this queue, so the only thing left to do is
    // evaluate the base case.
    const double baseCase = rule.BaseCase(queryIndex, point);
    Log::Debug << "Base case between " << queryIndex << " and " << point <<
        " evaluates to " << baseCase << ".\n";
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
