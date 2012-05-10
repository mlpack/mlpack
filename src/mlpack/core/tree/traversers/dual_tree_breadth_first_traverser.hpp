/**
 * @file dual_tree_breadth_first_traverser.hpp
 * @author Ryan Curtin
 *
 * A meta-heuristic-type class (is that even a word?) which allows an automatic
 * way to perform a breadth-first dual-tree traversal of two trees.  The trees
 * must be the same type.
 */
#ifndef __MLPACK_CORE_TREE_TRAVERSERS_DUAL_TREE_BREADTH_FIRST_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_TRAVERSERS_DUAL_TREE_BREADTH_FIRST_TRAVERSER_HPP

#include <mlpack/core.hpp>
#include <queue>

namespace mlpack {
namespace tree {

template<typename TreeType, typename RuleType>
class DualTreeBreadthFirstTraverser
{
 public:
  DualTreeBreadthFirstTraverser(RuleType& rule) :
      rule(rule), numPrunes(0)
  { /* Nothing to do. */ }

  void Traverse(TreeType& queryNode, TreeType& referenceNode)
  {
    // Each of these queues should have the same number of elements in them.
    std::queue<TreeType*> queryQueue;
    std::queue<TreeType*> referenceQueue;
    queryQueue.push(&queryNode);
    referenceQueue.push(&referenceNode);

    while (!queryQueue.empty())
    {
      TreeType& query = *queryStack.front();
      TreeType& reference = *referenceStack.front();
      queryStack.pop();
      referenceStack.pop();

      // Check if pruning can occur.
      if (rule.CanPrune(query, reference))
      {
        numPrunes++;
        continue;
      }

      // Run the base case for any points.  We must have points in both trees.
      if ((query.NumPoints() > 0) && (reference.NumPoints() > 0))
      {
        for (size_t i = 0; i < query.NumPoints(); ++i)
          for (size_t j = 0; j < reference.NumPoints(); ++j)
            rule.BaseCase(query.Point(i), reference.Point(j));
      }

      // Find the correct way to recurse given these two points.
      arma::Mat<size_t> recursionOrder;
      bool queryRecurse;
      bool referenceRecurse;
      rule.RecursionOrder(query, reference, recursionOrder, queryRecurse,
          referenceRecurse);

      // The RuleType has told us what we need to recurse on; act accordingly.
      if (queryRecurse && referenceRecurse) // Recurse on both trees.
      {
        for (size_t i = 0; i < recursionOrder.n_cols; ++i)
        {
          queryQueue.push(&(query.Child(recursionOrder(0, i))));
          referenceQueue.push(&(reference.Child(recursionOrder(1, i))));
        }
      }
      else if (queryRecurse) // Only recurse on query tree.
      {
        for (size_t i = 0; i < recursionOrder.n_cols; ++i)
        {
          queryQueue.push(&(query.Child(recursionOrder(0, i))));
          referenceQueue.push(&reference);
        }
      }
      else if (referenceRecurse) // Only recurse on reference tree.
      {
        for (size_t i = 0; i < recursionOrder.n_cols; ++i)
        {
          queryQueue.push(&query);
          referenceQueue.push(&(reference.Child(recursionOrder(1, i))));
        }
      }

      // Now, after recursion, update any information.
      rule.UpdateAfterRecursion(query, reference);
    }
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
