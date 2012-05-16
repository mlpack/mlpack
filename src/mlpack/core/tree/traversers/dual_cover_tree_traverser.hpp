/**
 * @file dual_cover_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 */
#ifndef __MLPACK_CORE_TREE_DUAL_COVER_TREE_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_DUAL_COVER_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>
#include <queue>

namespace mlpack {
namespace tree {

template<typename TreeType, typename RuleType>
class DualCoverTreeTraverser
{
 public:
  DualCoverTreeTraverser(RuleType& rule) : rule(rule), numPrunes(0) { }

  void Traverse(TreeType& queryNode, TreeType& referenceNode)
  {
    Traverse(queryNode, referenceNode, size_t() - 1);
  }

  void Traverse(TreeType& queryNode, TreeType& referenceNode, size_t parent)
  {
    std::queue<TreeType*> referenceQueue;
    std::queue<size_t> referenceParents;

    referenceQueue.push(&referenceNode);
    referenceParents.push(parent);

    while (!referenceQueue.empty())
    {
      TreeType& reference = *referenceQueue.front();
      referenceQueue.pop();

      size_t refParent = referenceParents.front();
      referenceParents.pop();

      // Do the base case, if we need to.
      if (refParent != reference.Point())
        rule.BaseCase(queryNode.Point(), reference.Point());

      if (((queryNode.Scale() < reference.Scale()) &&
           (reference.NumChildren() != 0)) ||
           (queryNode.NumChildren() == 0))
      {
        // We must descend the reference node.  Pruning happens here.
        for (size_t i = 0; i < reference.NumChildren(); ++i)
        {
          // Can we prune?
  //        if (!rule.CanPrune(queryNode, reference.Child(i)))
          {
            referenceQueue.push(&(reference.Child(i)));
            referenceParents.push(reference.Point());
          }
//          else
          {
            numPrunes++;
          }
        }
      }
      else
      {
        // We must descend the query node.  No pruning happens here.  For the
        // self-child, we trick the recursion into thinking that the base case
        // has already been done (which it has).
        if (queryNode.NumChildren() >= 1)
          Traverse(queryNode.Child(0), reference, reference.Point());

        for (size_t i = 1; i < queryNode.NumChildren(); ++i)
          Traverse(queryNode.Child(i), reference, size_t() - 1);
      }
    }
  }

  size_t NumPrunes() const { return numPrunes; }
  size_t& NumPrunes() { return numPrunes; }

 private:
  RuleType& rule;
  size_t numPrunes;
};

}; // namespace tree
}; // namespace mlpack

#endif
