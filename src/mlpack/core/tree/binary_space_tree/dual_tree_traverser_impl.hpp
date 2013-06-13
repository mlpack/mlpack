/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the DualTreeTraverser for BinarySpaceTree.  This is a way
 * to perform a dual-tree traversal of two trees.  The trees must be the same
 * type.
 *
 * This file is part of MLPACK 1.0.6.
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
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_traverser.hpp"

namespace mlpack {
namespace tree {

template<typename BoundType, typename StatisticType, typename MatType>
template<typename RuleType>
BinarySpaceTree<BoundType, StatisticType, MatType>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename BoundType, typename StatisticType, typename MatType>
template<typename RuleType>
void BinarySpaceTree<BoundType, StatisticType, MatType>::
DualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<BoundType, StatisticType, MatType>& queryNode,
    BinarySpaceTree<BoundType, StatisticType, MatType>& referenceNode)
{
  // If both are leaves, we must evaluate the base case.
  if (queryNode.IsLeaf() && referenceNode.IsLeaf())
  {
    // Loop through each of the points in each node.
    for (size_t query = queryNode.Begin(); query < queryNode.End(); ++query)
    {
      // See if we need to investigate this point (this function should be
      // implemented for the single-tree recursion too).
      const double score = rule.Score(query, referenceNode);

      if (score == DBL_MAX)
        continue; // We can't improve this particular point.

      for (size_t ref = referenceNode.Begin(); ref < referenceNode.End(); ++ref)
        rule.BaseCase(query, ref);
    }
  }
  else if ((!queryNode.IsLeaf()) && referenceNode.IsLeaf())
  {
    // We have to recurse down the query node.  In this case the recursion order
    // does not matter.
    double leftScore = rule.Score(*queryNode.Left(), referenceNode);

    if (leftScore != DBL_MAX)
      Traverse(*queryNode.Left(), referenceNode);
    else
      ++numPrunes;

    double rightScore = rule.Score(*queryNode.Right(), referenceNode);

    if (rightScore != DBL_MAX)
      Traverse(*queryNode.Right(), referenceNode);
    else
      ++numPrunes;
  }
  else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
  {
    // We have to recurse down the reference node.  In this case the recursion
    // order does matter.
    double leftScore = rule.Score(queryNode, *referenceNode.Left());
    double rightScore = rule.Score(queryNode, *referenceNode.Right());

    if (leftScore < rightScore)
    {
      // Recurse to the left.
      Traverse(queryNode, *referenceNode.Left());

      // Is it still valid to recurse to the right?
      rightScore = rule.Rescore(queryNode, *referenceNode.Right(), rightScore);

      if (rightScore != DBL_MAX)
        Traverse(queryNode, *referenceNode.Right());
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
        Traverse(queryNode, *referenceNode.Left());
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
        // Choose the left first.
        Traverse(queryNode, *referenceNode.Left());

        rightScore = rule.Rescore(queryNode, *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
          Traverse(queryNode, *referenceNode.Right());
        else
          ++numPrunes;
      }
    }
  }
  else
  {
    // We have to recurse down both query and reference nodes.  Because the
    // query descent order does not matter, we will go to the left query child
    // first.
    double leftScore = rule.Score(*queryNode.Left(), *referenceNode.Left());
    double rightScore = rule.Score(*queryNode.Left(), *referenceNode.Right());

    if (leftScore < rightScore)
    {
      // Recurse to the left.
      Traverse(*queryNode.Left(), *referenceNode.Left());

      // Is it still valid to recurse to the right?
      rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
          rightScore);

      if (rightScore != DBL_MAX)
        Traverse(*queryNode.Left(), *referenceNode.Right());
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
        Traverse(*queryNode.Left(), *referenceNode.Left());
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
        // Choose the left first.
        Traverse(*queryNode.Left(), *referenceNode.Left());

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
          Traverse(*queryNode.Left(), *referenceNode.Right());
        else
          ++numPrunes;
      }
    }

    // Now recurse down the right query node.
    leftScore = rule.Score(*queryNode.Right(), *referenceNode.Left());
    rightScore = rule.Score(*queryNode.Right(), *referenceNode.Right());

    if (leftScore < rightScore)
    {
      // Recurse to the left.
      Traverse(*queryNode.Right(), *referenceNode.Left());

      // Is it still valid to recurse to the right?
      rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
          rightScore);

      if (rightScore != DBL_MAX)
        Traverse(*queryNode.Right(), *referenceNode.Right());
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
        Traverse(*queryNode.Right(), *referenceNode.Left());
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
        // Choose the left first.
        Traverse(*queryNode.Right(), *referenceNode.Left());

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
          Traverse(*queryNode.Right(), *referenceNode.Right());
        else
          ++numPrunes;
      }
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

