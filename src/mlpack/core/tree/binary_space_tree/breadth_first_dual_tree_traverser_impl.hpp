/**
 * @file breadth_first_dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the BreadthFirstDualTreeTraverser for BinarySpaceTree.
 * This is a way to perform a dual-tree traversal of two trees.  The trees must
 * be the same type.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP

// In case it hasn't been included yet.
#include "breadth_first_dual_tree_traverser.hpp"

#include <queue>

namespace mlpack {
namespace tree {

template<typename BoundType,
         typename StatisticType,
         typename MatType,
         typename SplitType>
template<typename RuleType>
BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>::
BreadthFirstDualTreeTraverser<RuleType>::BreadthFirstDualTreeTraverser(
    RuleType& rule) :
    rule(rule),
    numPrunes(0),
    numVisited(0),
    numScores(0),
    numBaseCases(0)
{ /* Nothing to do. */ }

template<typename BoundType,
         typename StatisticType,
         typename MatType,
         typename SplitType>
template<typename RuleType>
void BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>::
BreadthFirstDualTreeTraverser<RuleType>::Traverse(
    BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>& queryRoot,
    BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>&
        referenceRoot)
{
  // Increment the visit counter.
  ++numVisited;

  // Store the current traversal info.
  traversalInfo = rule.TraversalInfo();

  typedef BinarySpaceTree<BoundType, StatisticType, MatType, SplitType>
      TreeType;

  std::queue<TreeType*> queryList;
  std::queue<TreeType*> referenceList;
  std::queue<typename RuleType::TraversalInfoType> traversalInfos;
  queryList.push(&queryRoot);
  referenceList.push(&referenceRoot);
  traversalInfos.push(rule.TraversalInfo());

  while (!queryList.empty())
  {
    TreeType& queryNode = *queryList.front();
    TreeType& referenceNode = *referenceList.front();
    typename RuleType::TraversalInfoType ti = traversalInfos.front();

    queryList.pop();
    referenceList.pop();
    traversalInfos.pop();

    rule.TraversalInfo() = ti;

    // If both are leaves, we must evaluate the base case.
    if (queryNode.IsLeaf() && referenceNode.IsLeaf())
    {
      // Loop through each of the points in each node.
      for (size_t query = queryNode.Begin(); query < queryNode.End(); ++query)
      {
        // See if we need to investigate this point (this function should be
        // implemented for the single-tree recursion too).  Restore the traversal
        // information first.
//        const double childScore = rule.Score(query, referenceNode);

//        if (childScore == DBL_MAX)
//          continue; // We can't improve this particular point.

        for (size_t ref = referenceNode.Begin(); ref < referenceNode.End(); ++ref)
          rule.BaseCase(query, ref);

        numBaseCases += referenceNode.Count();
      }
    }
    else if ((!queryNode.IsLeaf()) && referenceNode.IsLeaf())
    {
      // We have to recurse down the query node.  In this case the recursion order
      // does not matter.
      const double leftScore = rule.Score(*queryNode.Left(), referenceNode);
      ++numScores;

      if (leftScore != DBL_MAX)
      {
        queryList.push(queryNode.Left());
        referenceList.push(&referenceNode);
        traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push1 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
      }
      else
      {
        ++numPrunes;
      }

      // Before recursing, we have to set the traversal information correctly.
      rule.TraversalInfo() = ti;
      const double rightScore = rule.Score(*queryNode.Right(), referenceNode);
      ++numScores;

      if (rightScore != DBL_MAX)
      {
        queryList.push(queryNode.Right());
        referenceList.push(&referenceNode);
        traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push2 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
      }
      else
        ++numPrunes;
    }
    else if (queryNode.IsLeaf() && (!referenceNode.IsLeaf()))
    {
      // We have to recurse down the reference node.  In this case the recursion
      // order does matter.  Before recursing, though, we have to set the
      // traversal information correctly.
      double leftScore = rule.Score(queryNode, *referenceNode.Left());
      typename RuleType::TraversalInfoType leftInfo = rule.TraversalInfo();
      rule.TraversalInfo() = ti;
      double rightScore = rule.Score(queryNode, *referenceNode.Right());
      numScores += 2;

      if (leftScore < rightScore)
      {
        // Recurse to the left.  Restore the left traversal info.  Store the right
        // traversal info.
        queryList.push(&queryNode);
        referenceList.push(referenceNode.Left());
        traversalInfos.push(leftInfo);
//        Log::Debug << "Push3 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(queryNode, *referenceNode.Right(), rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          queryList.push(&queryNode);
          referenceList.push(referenceNode.Right());
          traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push4 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
        }
        else
          ++numPrunes;
      }
      else if (rightScore < leftScore)
    {
      // Recurse to the right.
      queryList.push(&queryNode);
      referenceList.push(referenceNode.Right());
      traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push5 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

      // Is it still valid to recurse to the left?
      leftScore = rule.Rescore(queryNode, *referenceNode.Left(), leftScore);

      if (leftScore != DBL_MAX)
      {
        // Restore the left traversal info.
        queryList.push(&queryNode);
        referenceList.push(referenceNode.Left());
        traversalInfos.push(leftInfo);
//        Log::Debug << "Push6 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
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
        queryList.push(&queryNode);
        referenceList.push(referenceNode.Left());
        traversalInfos.push(leftInfo);
//        Log::Debug << "Push7 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

        rightScore = rule.Rescore(queryNode, *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          queryList.push(&queryNode);
          referenceList.push(referenceNode.Right());
          traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push8 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
        }
        else
          ++numPrunes;
      }
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
    rule.TraversalInfo() = ti;
    double rightScore = rule.Score(*queryNode.Left(), *referenceNode.Right());
    typename RuleType::TraversalInfoType rightInfo;
    numScores += 2;

    if (leftScore < rightScore)
    {
      // Recurse to the left.  Restore the left traversal info.  Store the right
      // traversal info.
      queryList.push(queryNode.Left());
      referenceList.push(referenceNode.Left());
      traversalInfos.push(leftInfo);
//        Log::Debug << "Push9 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

      // Is it still valid to recurse to the right?
      rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
          rightScore);

      if (rightScore != DBL_MAX)
      {
        // Restore the right traversal info.
        queryList.push(queryNode.Left());
        referenceList.push(referenceNode.Right());
        traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push10 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
      }
      else
        ++numPrunes;
    }
    else if (rightScore < leftScore)
    {
      // Recurse to the right.
      queryList.push(queryNode.Left());
      referenceList.push(referenceNode.Right());
      traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push11 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

      // Is it still valid to recurse to the left?
      leftScore = rule.Rescore(*queryNode.Left(), *referenceNode.Left(),
          leftScore);

      if (leftScore != DBL_MAX)
      {
        // Restore the left traversal info.
        queryList.push(queryNode.Left());
        referenceList.push(referenceNode.Left());
        traversalInfos.push(leftInfo);
//        Log::Debug << "Push12 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
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
        // Choose the left first.  Restore the left traversal info and store the
        // right traversal info.
        queryList.push(queryNode.Left());
        referenceList.push(referenceNode.Left());
        traversalInfos.push(leftInfo);
//        Log::Debug << "Push13 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Left(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal information.
          queryList.push(queryNode.Left());
          referenceList.push(referenceNode.Right());
          traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push14 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
        }
        else
          ++numPrunes;
      }
    }

    // Restore the main traversal information.
    rule.TraversalInfo() = ti;

    // Now recurse down the right query node.
    leftScore = rule.Score(*queryNode.Right(), *referenceNode.Left());
    leftInfo = rule.TraversalInfo();
    rule.TraversalInfo() = ti;
    rightScore = rule.Score(*queryNode.Right(), *referenceNode.Right());
    numScores += 2;

    if (leftScore < rightScore)
    {
      // Recurse to the left.  Restore the left traversal info.  Store the right
      // traversal info.
      queryList.push(queryNode.Right());
      referenceList.push(referenceNode.Left());
      traversalInfos.push(leftInfo);
//        Log::Debug << "Push15 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

      // Is it still valid to recurse to the right?
      rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
          rightScore);

      if (rightScore != DBL_MAX)
      {
        // Restore the right traversal info.
        queryList.push(queryNode.Right());
        referenceList.push(referenceNode.Right());
        traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push16 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
      }
      else
        ++numPrunes;
    }
    else if (rightScore < leftScore)
    {
      // Recurse to the right.
      queryList.push(queryNode.Right());
      referenceList.push(referenceNode.Right());
      traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push17 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

      // Is it still valid to recurse to the left?
      leftScore = rule.Rescore(*queryNode.Right(), *referenceNode.Left(),
          leftScore);

      if (leftScore != DBL_MAX)
      {
        // Restore the left traversal info.
        queryList.push(queryNode.Right());
        referenceList.push(referenceNode.Left());
        traversalInfos.push(leftInfo);
//        Log::Debug << "Push18 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
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
        queryList.push(queryNode.Right());
        referenceList.push(referenceNode.Left());
        traversalInfos.push(leftInfo);
//        Log::Debug << "Push19 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";

        // Is it still valid to recurse to the right?
        rightScore = rule.Rescore(*queryNode.Right(), *referenceNode.Right(),
            rightScore);

        if (rightScore != DBL_MAX)
        {
          // Restore the right traversal info.
          queryList.push(queryNode.Right());
          referenceList.push(referenceNode.Right());
          traversalInfos.push(rule.TraversalInfo());
//        Log::Debug << "Push20 " << queryList.back()->Begin() << ", " <<
//queryList.back()->Count() << "; " << referenceList.back()->Begin() << ", "
//    << referenceList.back()->Count() << "\n";
        }
        else
          ++numPrunes;
      }
    }
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BREADTH_FIRST_DUAL_TREE_TRAVERSER_IMPL_HPP
