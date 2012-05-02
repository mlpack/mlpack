/**
 * @file single_tree_depth_first_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the SingleTreeDepthFirstTraverser, an extensible class which allows a
 * single-tree depth-first recursion with a pruning rule and a base case
 * (two point) rule.
 */
#ifndef __MLPACK_CORE_TREE_TRAVERSERS_SINGLE_TREE_DEPTH_FIRST_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_TRAVERSERS_SINGLE_TREE_DEPTH_FIRST_TRAVERSER_HPP

#include <mlpack/core.hpp>
#include <stack>

namespace mlpack {
namespace tree {

template<typename TreeType, typename RuleType>
class SingleTreeDepthFirstTraverser
{
 public:
  SingleTreeDepthFirstTraverser(RuleType& rule) :
      rule(rule),
      numPrunes(0)
  { /* Nothing to do. */ }

  void Traverse(const size_t queryIndex, TreeType& referenceNode)
  {
    // This is a non-recursive implementation (which should be faster).

    // The stack of points to look at.
    std::stack<TreeType*> pointStack;
    pointStack.push(&referenceNode);

    while (!pointStack.empty())
    {
      TreeType* node = pointStack.top();
      pointStack.pop();

      // Check if we can prune this node.
      if (rule.CanPrune(queryIndex, *node))
      {
        ++numPrunes;
        continue;
      }

      // First run the base case for any points this node might hold.
      for (size_t i = 0; i < node->NumPoints(); ++i)
        rule.BaseCase(queryIndex, node->Point(i));

      for (size_t i = 0; i < node->NumChildren(); ++i)
        pointStack.push(&(node->Child(i)));
    }
  }

  //! Get the number of prunes so far.
  size_t NumPrunes() const { return numPrunes; }
  //! Set the number of prunes (good for a reset to 0).
  size_t& NumPrunes() { return numPrunes; }

 private:
  RuleType& rule;

  size_t numPrunes;
};

}; // namespace tree
}; // namespace mlpack

#endif
