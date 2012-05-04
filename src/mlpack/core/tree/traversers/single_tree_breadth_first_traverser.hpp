/**
 * @file single_tree_breadth_first_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the SingleTreeBreadthFirstTraverser, an extensible class which allows
 * a single-tree breadth-first recursion with a pruning rule and a base case
 * (two point) rule.  This is similar to the SingleTreeDepthFirstTraverser.
 */
#ifndef __MLPACK_CORE_TREE_TRAVERSERS_SINGLE_TREE_BREADTH_FIRST_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_TRAVERSERS_SINGLE_TREE_BREADTH_FIRST_TRAVERSER_HPP

#include <mlpack/core.hpp>
#include <queue>

namespace mlpack {
namespace tree {

template<typename TreeType, typename RuleType>
class SingleTreeBreadthFirstTraverser
{
 public:
  SingleTreeBreadthFirstTraverser(RuleType& rule) :
      rule(rule),
      numPrunes(0)
  { /* Nothing to do. */ }

  void Traverse(const size_t queryIndex, TreeType& referenceNode)
  {
    // This is a non-recursive implementation (which should be faster than a
    // recursive implementation).
    std::queue<TreeType*> pointQueue;
    std::queue<size_t> parentPoints; // For if this tree has self-children.
    pointQueue.push(&referenceNode);

    if (TreeType::HasSelfChildren())
      parentPoints.push(size_t() - 1); // Invalid value.

    while (!pointQueue.empty())
    {
      TreeType* node = pointQueue.front();
      pointQueue.pop();

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
      size_t baseCaseStart = 0;
      if (TreeType::HasSelfChildren())
      {
        if (parentPoints.front() == node->Point(0))
          baseCaseStart = 1; // Skip base case we've already evaluated.

        parentPoints.pop();
      }

      // First run the base case for any points this node might hold.
      for (size_t i = baseCaseStart; i < node->NumPoints(); ++i)
        rule.BaseCase(queryIndex, node->Point(i));

      // Now push children into the FIFO.
      for (size_t i = 0; i < node->NumChildren(); ++i)
        pointQueue.push(&(node->Child(i)));

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
