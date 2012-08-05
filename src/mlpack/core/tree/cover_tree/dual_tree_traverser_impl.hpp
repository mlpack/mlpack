/**
 * @file dual_tree_traverser_impl.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_IMPL_HPP

#include <mlpack/core.hpp>
#include <queue>

namespace mlpack {
namespace tree {

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::DualTreeTraverser(RuleType& rule) :
    rule(rule),
    numPrunes(0)
{ /* Nothing to do. */ }

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::Traverse(
    CoverTree<MetricType, RootPointPolicy, StatisticType>& queryNode,
    CoverTree<MetricType, RootPointPolicy, StatisticType>& referenceNode)
{
  // Start traversal with an invalid parent index.
  Traverse(queryNode, referenceNode, size_t() - 1);
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::
DualTreeTraverser<RuleType>::Traverse(
  CoverTree<MetricType, RootPointPolicy, StatisticType>& queryNode,
  CoverTree<MetricType, RootPointPolicy, StatisticType>& referenceNode,
  const size_t parent)
{
  std::queue<CoverTree<MetricType, RootPointPolicy, StatisticType>*>
      referenceQueue;
  std::queue<size_t> referenceParents;

  referenceQueue.push(&referenceNode);
  referenceParents.push(parent);

  while (!referenceQueue.empty())
  {
    CoverTree<MetricType, RootPointPolicy, StatisticType>& reference =
        *referenceQueue.front();
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
        if (!rule.CanPrune(queryNode, reference.Child(i)))
        {
          referenceQueue.push(&(reference.Child(i)));
          referenceParents.push(reference.Point());
        }
        else
        {
          ++numPrunes;
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

}; // namespace tree
}; // namespace mlpack

#endif
