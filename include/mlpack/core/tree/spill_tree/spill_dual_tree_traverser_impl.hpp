/**
 * @file core/tree/spill_tree/spill_dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Implementation of the SpillDualTreeTraverser for SpillTree.  This is a way
 * to perform a dual-tree traversal of two trees.  The trees must be the same
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPILL_DUAL_TREE_TRAVERSER_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPILL_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "spill_dual_tree_traverser.hpp"

namespace mlpack {

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneDistanceType> class HyperplaneType,
         template<typename SplitDistanceType, typename SplitMatType>
             class SplitType>
template<typename RuleType, bool Defeatist>
SpillTree<DistanceType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillDualTreeTraverser<RuleType, Defeatist>::SpillDualTreeTraverser(
    RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do. */ }

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneDistanceType> class HyperplaneType,
         template<typename SplitDistanceType, typename SplitMatType>
             class SplitType>
template<typename RuleType, bool Defeatist>
void
SpillTree<DistanceType, StatisticType, MatType, HyperplaneType, SplitType>::
SpillDualTreeTraverser<RuleType, Defeatist>::Traverse(
    SpillTree<DistanceType, StatisticType, MatType, HyperplaneType, SplitType>&
        queryNode,
    SpillTree<DistanceType, StatisticType, MatType, HyperplaneType, SplitType>&
        referenceNode,
    const bool bruteForce)
{
  // Increment the visit counter.
  ++numVisited;

  // Store the current traversal info.
  traversalInfo = rule.TraversalInfo();

  // Determine whether we need to brute-force the reference node.  We have no
  // realistic way to track how many base cases we've done for each point, so we
  // act as though we have done zero.
  if (!bruteForce && Defeatist &&
      (referenceNode.Parent() != NULL) &&
      (referenceNode.Parent()->Overlap()) &&
      (referenceNode.NumDescendants() < rule.MinimumBaseCases()))
  {
    // We've actually recursed too far.  Go back up one level and brute-force
    // the computation, and then we are done.
    Traverse(queryNode, *referenceNode.Parent(), true);
    return;
  }
  else if ((queryNode.IsLeaf() && referenceNode.IsLeaf()) || bruteForce)
  {
    // If both are leaves or if we explicitly need to do brute-force search, we
    // must evaluate the base cases.

    // Loop through each of the points in each node.
    const size_t queryEnd = queryNode.NumDescendants();
    const size_t refEnd = referenceNode.NumDescendants();
    for (size_t query = 0; query < queryEnd; ++query)
    {
      const size_t queryIndex = queryNode.Descendant(query);
      // See if we need to investigate this point.  Restore the traversal
      // information first.
      rule.TraversalInfo() = traversalInfo;
      const double childScore = rule.Score(queryIndex, referenceNode);

      if (childScore == DBL_MAX)
        continue; // We can't improve this particular point.

      for (size_t ref = 0; ref < refEnd; ++ref)
        rule.BaseCase(queryIndex, referenceNode.Descendant(ref));

      numBaseCases += refEnd;
    }
  }
  else if (((!queryNode.IsLeaf()) && referenceNode.IsLeaf()) ||
           (queryNode.NumDescendants() > 3 * referenceNode.NumDescendants() &&
            !queryNode.IsLeaf() && !referenceNode.IsLeaf()))
  {
    // We have to recurse down the query node.  In this case the recursion order
    // does not matter.
    const double leftScore = rule.Score(*queryNode.Left(), referenceNode);
    ++numScores;

    if (leftScore != DBL_MAX)
      Traverse(*queryNode.Left(), referenceNode);
    else
      ++numPrunes;

    // Before recursing, we have to set the traversal information correctly.
    rule.TraversalInfo() = traversalInfo;
    const double rightScore = rule.Score(*queryNode.Right(), referenceNode);
    ++numScores;

    if (rightScore != DBL_MAX)
      Traverse(*queryNode.Right(), referenceNode);
    else
      ++numPrunes;
  }
  else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
  {
    if (Defeatist && referenceNode.Overlap())
    {
      // If referenceNode is a overlapping node let's do defeatist search.
      size_t bestChild = rule.GetBestChild(queryNode, referenceNode);
      if (bestChild < referenceNode.NumChildren())
      {
        Traverse(queryNode, referenceNode.Child(bestChild));
        ++numPrunes;
      }
      else
      {
        // If we can't decide which child node to traverse, this means that
        // queryNode is at both sides of the splitting hyperplane. So, as
        // queryNode is a leafNode, all we can do is single tree search for each
        // point in the query node.
        const size_t queryEnd = queryNode.NumPoints();
        DefeatistSingleTreeTraverser<RuleType> st(rule);
        // Loop through each of the points in query node.
        for (size_t query = 0; query < queryEnd; ++query)
        {
          const size_t queryIndex = queryNode.Point(query);
          // See if we need to investigate this point.
          const double childScore = rule.Score(queryIndex, referenceNode);

          if (childScore == DBL_MAX)
            continue; // We can't improve this particular point.

          st.Traverse(queryIndex, referenceNode);
        }
      }
    }
    else
    {
      // We have to recurse down the reference node.  In this case the recursion
      // order does matter.  Before recursing, though, we have to set the
      // traversal information correctly.
      double leftScore = rule.Score(queryNode, *referenceNode.Left());
      typename RuleType::TraversalInfoType leftInfo = rule.TraversalInfo();
      rule.TraversalInfo() = traversalInfo;
      double rightScore = rule.Score(queryNode, *referenceNode.Right());
      numScores += 2;

      if (leftScore < rightScore)
      {
        // Recurse to the left.  Restore the left traversal info.  Store the
        // right traversal info.
        traversalInfo = rule.TraversalInfo();
        rule.TraversalInfo() = leftInfo;
        Traverse(queryNode, *referenceNode.Left());

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(queryNode, *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          rule.TraversalInfo() = traversalInfo;
          Traverse(queryNode, *referenceNode.Right());
        }
        else
          ++numPrunes;
      }
      else if (rightScore < leftScore)
      {
        // Recurse to the right.
        Traverse(queryNode, *referenceNode.Right());

        // Is it still valid to recurse to the left?
        leftScore = rule.Rescore(queryNode, *referenceNode.Left(), leftScore);

        if (leftScore != DBL_MAX)
        {
          // Restore the left traversal info.
          rule.TraversalInfo() = leftInfo;
          Traverse(queryNode, *referenceNode.Left());
        }
        else
          ++numPrunes;
      }
      else // leftScore is equal to rightScore.
      {
        if (leftScore == DBL_MAX)
        {
          numPrunes += 2;
        }
        else
        {
          // Choose the left first.  Restore the left traversal info.  Store the
          // right traversal info.
          traversalInfo = rule.TraversalInfo();
          rule.TraversalInfo() = leftInfo;
          Traverse(queryNode, *referenceNode.Left());

          rightScore = rule.Rescore(queryNode, *referenceNode.Right(),
              rightScore);

          if (rightScore != DBL_MAX)
          {
            // Restore the right traversal info.
            rule.TraversalInfo() = traversalInfo;
            Traverse(queryNode, *referenceNode.Right());
          }
          else
            ++numPrunes;
        }
      }
    }
  }
  else
  {
    if (Defeatist && referenceNode.Overlap())
    {
      // If referenceNode is a overlapping node let's do defeatist search.
      size_t bestChild = rule.GetBestChild(*queryNode.Left(), referenceNode);
      if (bestChild < referenceNode.NumChildren())
      {
        Traverse(*queryNode.Left(), referenceNode.Child(bestChild));
        ++numPrunes;
      }
      else
      {
        // If we can't decide which child node to traverse, this means that
        // queryNode.Left() is at both sides of the splitting hyperplane. So,
        // let's recurse down only the query node.
        Traverse(*queryNode.Left(), referenceNode);
      }

      bestChild = rule.GetBestChild(*queryNode.Right(), referenceNode);
      if (bestChild < referenceNode.NumChildren())
      {
        Traverse(*queryNode.Right(), referenceNode.Child(bestChild));
        ++numPrunes;
      }
      else
      {
        // If we can't decide which child node to traverse, this means that
        // queryNode.Right() is at both sides of the splitting hyperplane. So,
        // let's recurse down only the query node.
        Traverse(*queryNode.Right(), referenceNode);
      }
    }
    else
    {
      // We have to recurse down both query and reference nodes.  Because the
      // query descent order does not matter, we will go to the left query child
      // first.  Before recursing, we have to set the traversal information
      // correctly.
      double leftScore = rule.Score(*queryNode.Left(), *referenceNode.Left());
      typename RuleType::TraversalInfoType leftInfo = rule.TraversalInfo();
      rule.TraversalInfo() = traversalInfo;
      double rightScore = rule.Score(*queryNode.Left(), *referenceNode.Right());
      typename RuleType::TraversalInfoType rightInfo;
      numScores += 2;

      if (leftScore < rightScore)
      {
        // Recurse to the left.  Restore the left traversal info.  Store the
        // right traversal info.
        rightInfo = rule.TraversalInfo();
        rule.TraversalInfo() = leftInfo;
        Traverse(*queryNode.Left(), *referenceNode.Left());

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          rule.TraversalInfo() = rightInfo;
          Traverse(*queryNode.Left(), *referenceNode.Right());
        }
        else
          ++numPrunes;
      }
      else if (rightScore < leftScore)
      {
        // Recurse to the right.
        Traverse(*queryNode.Left(), *referenceNode.Right());

        // Is it still valid to recurse to the left?
        leftScore = rule.Rescore(*queryNode.Left(), *referenceNode.Left(),
            leftScore);

        if (leftScore != DBL_MAX)
        {
          // Restore the left traversal info.
          rule.TraversalInfo() = leftInfo;
          Traverse(*queryNode.Left(), *referenceNode.Left());
        }
        else
          ++numPrunes;
      }
      else
      {
        if (leftScore == DBL_MAX)
        {
          numPrunes += 2;
        }
        else
        {
          // Choose the left first.  Restore the left traversal info and store
          // the right traversal info.
          rightInfo = rule.TraversalInfo();
          rule.TraversalInfo() = leftInfo;
          Traverse(*queryNode.Left(), *referenceNode.Left());

          // Is it still valid to recurse to the right?
          rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
              rightScore);

          if (rightScore != DBL_MAX)
          {
            // Restore the right traversal information.
            rule.TraversalInfo() = rightInfo;
            Traverse(*queryNode.Left(), *referenceNode.Right());
          }
          else
            ++numPrunes;
        }
      }

      // Restore the main traversal information.
      rule.TraversalInfo() = traversalInfo;

      // Now recurse down the right query node.
      leftScore = rule.Score(*queryNode.Right(), *referenceNode.Left());
      leftInfo = rule.TraversalInfo();
      rule.TraversalInfo() = traversalInfo;
      rightScore = rule.Score(*queryNode.Right(), *referenceNode.Right());
      numScores += 2;

      if (leftScore < rightScore)
      {
        // Recurse to the left.  Restore the left traversal info.  Store the
        // right traversal info.
        rightInfo = rule.TraversalInfo();
        rule.TraversalInfo() = leftInfo;
        Traverse(*queryNode.Right(), *referenceNode.Left());

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          rule.TraversalInfo() = rightInfo;
          Traverse(*queryNode.Right(), *referenceNode.Right());
        }
        else
          ++numPrunes;
      }
      else if (rightScore < leftScore)
      {
        // Recurse to the right.
        Traverse(*queryNode.Right(), *referenceNode.Right());

        // Is it still valid to recurse to the left?
        leftScore = rule.Rescore(*queryNode.Right(), *referenceNode.Left(),
            leftScore);

        if (leftScore != DBL_MAX)
        {
          // Restore the left traversal info.
          rule.TraversalInfo() = leftInfo;
          Traverse(*queryNode.Right(), *referenceNode.Left());
        }
        else
          ++numPrunes;
      }
      else
      {
        if (leftScore == DBL_MAX)
        {
          numPrunes += 2;
        }
        else
        {
          // Choose the left first.  Restore the left traversal info.  Store the
          // right traversal info.
          rightInfo = rule.TraversalInfo();
          rule.TraversalInfo() = leftInfo;
          Traverse(*queryNode.Right(), *referenceNode.Left());

          // Is it still valid to recurse to the right?
          rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
              rightScore);

          if (rightScore != DBL_MAX)
          {
            // Restore the right traversal info.
            rule.TraversalInfo() = rightInfo;
            Traverse(*queryNode.Right(), *referenceNode.Right());
          }
          else
            ++numPrunes;
        }
      }
    }
  }
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_SPILL_TREE_SPILL_DUAL_TREE_TRAVERSER_IMPL_HPP
