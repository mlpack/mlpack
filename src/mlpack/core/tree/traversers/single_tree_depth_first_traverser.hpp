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
    std::stack<size_t> parentPoints; // For if this tree has self-children.
    pointStack.push(&referenceNode);

    if (TreeType::HasSelfChildren())
      parentPoints.push(size_t() - 1); // Invalid value.

    while (!pointStack.empty())
    {
      TreeType* node = pointStack.top();
      pointStack.pop();

      // Check if we can prune this node.
      if (rule.CanPrune(queryIndex, *node))
      {
        if (TreeType::HasSelfChildren())
          parentPoints.pop(); // Pop the parent point off.

        ++numPrunes;
        continue;
      }

      // If this tree type has self-children, we need to make sure we don't run
      // the base case if the parent already had it run.
      Log::Debug << "Node is " << node->Point(0) << std::endl;
      size_t baseCaseStart = 0;
      if (TreeType::HasSelfChildren())
      {
        if (parentPoints.top() == node->Point(0))
          baseCaseStart = 1; // Skip base case we've already evaluated.

        parentPoints.pop();
      }

      // First run the base case for any points this node might hold.
      for (size_t i = baseCaseStart; i < node->NumPoints(); ++i)
        rule.BaseCase(queryIndex, node->Point(i));

      for (size_t i = 0; i < node->NumChildren(); ++i)
        pointStack.push(&(node->Child(i)));

      // Push parent points if we need to.
      if (TreeType::HasSelfChildren())
      {
        for (size_t i = 0; i < node->NumChildren(); ++i)
          parentPoints.push(node->Point(0));
      }
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
