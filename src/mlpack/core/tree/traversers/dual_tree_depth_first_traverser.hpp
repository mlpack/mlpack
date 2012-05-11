/**
 * @file dual_tree_depth_first_traverser.hpp
 * @author Ryan Curtin
 *
 * A meta-heuristic-type class (is that even a word?) which allows an automatic
 * way to perform a dual-tree traversal of two trees.  The trees must be the
 * same type.
 */
#ifndef __MLPACK_CORE_TREE_TRAVERSERS_DUAL_TREE_DEPTH_FIRST_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_TRAVERSERS_DUAL_TREE_DEPTH_FIRST_TRAVERSER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

template<typename TreeType, typename RuleType>
class DualTreeDepthFirstTraverser
{
 public:
  DualTreeDepthFirstTraverser(RuleType& rule) :
      rule(rule), numPrunes(0)
  { /* Nothing to do. */ }

  void Traverse(TreeType& queryNode, TreeType& referenceNode)
  {
    // Check if pruning can occur.
    if (rule.CanPrune(queryNode, referenceNode))
    {
      numPrunes++;
      return;
    }

    // Run the base case for any points.  We must have points in both trees.
    if ((queryNode.NumPoints() > 0) && (referenceNode.NumPoints() > 0))
    {
      for (size_t i = 0; i < queryNode.NumPoints(); ++i)
        for (size_t j = 0; j < referenceNode.NumPoints(); ++j)
          rule.BaseCase(queryNode.Point(i), referenceNode.Point(j));
    }

    // Find the correct way to recurse given these two points.
    arma::Mat<size_t> recursionOrder;
    bool queryRecurse;
    bool referenceRecurse;
    rule.RecursionOrder(queryNode, referenceNode, recursionOrder, queryRecurse,
                        referenceRecurse);

    // The RuleType has told us what we need to recurse on; act accordingly.
    if (queryRecurse && referenceRecurse) // Recurse on both trees.
    {
      for (size_t i = 0; i < recursionOrder.n_cols; ++i)
        Traverse(queryNode.Child(recursionOrder(0, i)),
                 referenceNode.Child(recursionOrder(1, i)));
    }
    else if (queryRecurse) // Only recurse on query tree.
    {
      for (size_t i = 0; i < recursionOrder.n_cols; ++i)
        Traverse(queryNode.Child(recursionOrder(0, i)), referenceNode);
    }
    else if (referenceRecurse) // Only recurse on reference tree.
    {
      for (size_t i = 0; i < recursionOrder.n_cols; ++i)
        Traverse(queryNode, referenceNode.Child(recursionOrder(1, i)));
    }

    // Now, after recursion, update any information.
    rule.UpdateAfterRecursion(queryNode, referenceNode);
  }

  size_t NumPrunes() const { return numPrunes; }
  size_t& NumPrunes() { return numPrunes; }

 private:
  RuleType& rule;
  size_t numPrunes;
};

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_TRAVERSERS_DUAL_TREE_DEPTH_FIRST_TRAVERSER_HPP

