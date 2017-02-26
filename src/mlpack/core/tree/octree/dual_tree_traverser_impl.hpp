/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the dual-tree traverser for the octree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_OCTREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_OCTREE_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType, typename StatisticType, typename MatType>
template<typename RuleType>
Octree<MetricType, StatisticType, MatType>::DualTreeTraverser<RuleType>::
    DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{
  // Nothing to do.
}

template<typename MetricType, typename StatisticType, typename MatType>
template<typename RuleType>
void Octree<MetricType, StatisticType, MatType>::DualTreeTraverser<RuleType>::
    Traverse(Octree& queryNode, Octree& referenceNode)
{
  // Increment the visit counter.
  ++numVisited;

  // Store the current traversal info.
  traversalInfo = rule.TraversalInfo();

  if (queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    const size_t begin = queryNode.Point(0);
    const size_t end = begin + queryNode.NumPoints();
    for (size_t q = begin; q < end; ++q)
    {
      // First, see if we can prune the reference node for this query point.
      rule.TraversalInfo() = traversalInfo;
      const double score = rule.Score(q, referenceNode);
      if (score == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      const size_t rBegin = referenceNode.Point(0);
      const size_t rEnd = rBegin + referenceNode.NumPoints();
      for (size_t r = rBegin; r < rEnd; ++r)
        rule.BaseCase(q, r);

      numBaseCases += referenceNode.NumPoints();
    }
  }
  else if (!queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    // We have to recurse down the query node.  Order does not matter.
    for (size_t i = 0; i < queryNode.NumChildren(); ++i)
    {
      rule.TraversalInfo() = traversalInfo;
      const double score = rule.Score(queryNode.Child(i), referenceNode);
      if (score == DBL_MAX)
      {
        ++numPrunes;
        continue;
      }

      Traverse(queryNode.Child(i), referenceNode);
    }
  }
  else if (queryNode.IsLeaf() && !referenceNode.IsLeaf())
  {
    // We have to recurse down the reference node, so we need to do it in an
    // ordered manner.
    arma::vec scores(referenceNode.NumChildren());
    std::vector<typename RuleType::TraversalInfoType>
        tis(referenceNode.NumChildren());
    for (size_t i = 0; i < referenceNode.NumChildren(); ++i)
    {
      rule.TraversalInfo() = traversalInfo;
      scores[i] = rule.Score(queryNode, referenceNode.Child(i));
      tis[i] = rule.TraversalInfo();
    }

    // Sort the scores.
    arma::uvec scoreOrder = arma::sort_index(scores);
    for (size_t i = 0; i < scoreOrder.n_elem; ++i)
    {
      if (scores[scoreOrder[i]] == DBL_MAX)
      {
        // We don't need to check any more---all children past here are pruned.
        numPrunes += scoreOrder.n_elem - i;
        break;
      }

      rule.TraversalInfo() = tis[scoreOrder[i]];
      Traverse(queryNode, referenceNode.Child(scoreOrder[i]));
    }
  }
  else
  {
    // We have to recurse down both the query and reference nodes.  Query order
    // does not matter, so we will do that in sequence.  However we will
    // allocate the arrays for recursion at this level.
    arma::vec scores(referenceNode.NumChildren());
    std::vector<typename RuleType::TraversalInfoType>
        tis(referenceNode.NumChildren());
    for (size_t j = 0; j < queryNode.NumChildren(); ++j)
    {
      // Now we have to recurse down the reference node, which we will do in a
      // prioritized manner.
      for (size_t i = 0; i < referenceNode.NumChildren(); ++i)
      {
        rule.TraversalInfo() = traversalInfo;
        scores[i] = rule.Score(queryNode.Child(j), referenceNode.Child(i));
        tis[i] = rule.TraversalInfo();
      }

      // Sort the scores.
      arma::uvec scoreOrder = arma::sort_index(scores);
      for (size_t i = 0; i < scoreOrder.n_elem; ++i)
      {
        if (scores[scoreOrder[i]] == DBL_MAX)
        {
          // We don't need to check any more---all children past here are pruned.
          numPrunes += scoreOrder.n_elem - i;
          break;
        }

        rule.TraversalInfo() = tis[scoreOrder[i]];
        Traverse(queryNode.Child(j), referenceNode.Child(scoreOrder[i]));
      }
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif
