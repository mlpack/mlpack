/**
 * @file minimal_splits_number_sweep_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the MinimalSplitsNumberSweep class, a class that finds a
 * partition of a node along an axis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_IMPL_HPP

#include "minimal_splits_number_sweep.hpp"

namespace mlpack {
namespace tree {

template<typename SplitPolicy>
template<typename TreeType>
size_t MinimalSplitsNumberSweep<SplitPolicy>::SweepNonLeafNode(
    const size_t axis,
    const TreeType* node,
    typename TreeType::ElemType& axisCut)
{
  typedef typename TreeType::ElemType ElemType;

  std::vector<std::pair<ElemType, size_t>> sorted(node->NumChildren());

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    sorted[i].first = SplitPolicy::Bound(node->Child(i))[axis].Hi();
    sorted[i].second = i;
  }

  // Sort candidates in order to check balancing.
  std::sort(sorted.begin(), sorted.end(),
      [] (const std::pair<ElemType, size_t>& s1,
          const std::pair<ElemType, size_t>& s2)
      {
        return s1.first < s2.first;
      });

  size_t minCost = SIZE_MAX;

  // Find a split with the minimal cost.
  for (size_t i = 0; i < sorted.size(); i++)
  {
    size_t numTreeOneChildren = 0;
    size_t numTreeTwoChildren = 0;
    size_t numSplits = 0;

    // Calculate the number of splits.
    for (size_t j = 0; j < node->NumChildren(); j++)
    {
      const TreeType& child = node->Child(j);
      int policy = SplitPolicy::GetSplitPolicy(child, axis, sorted[i].first);
      if (policy == SplitPolicy::AssignToFirstTree)
        numTreeOneChildren++;
      else if (policy == SplitPolicy::AssignToSecondTree)
        numTreeTwoChildren++;
      else
      {
        numTreeOneChildren++;
        numTreeTwoChildren++;
        numSplits++;
      }
    }

    // Check if the split is possible.
    if (numTreeOneChildren <= node->MaxNumChildren() && numTreeOneChildren > 0 &&
        numTreeTwoChildren <= node->MaxNumChildren() && numTreeTwoChildren > 0)
    {
      // Evaluate the cost using the number of splits and balancing.
      size_t balance;

      if (sorted.size() / 2 > i )
        balance = sorted.size() / 2 - i;
      else
        balance = i - sorted.size() / 2;

      size_t cost = numSplits * balance;
      if (cost < minCost)
      {
        minCost = cost;
        axisCut = sorted[i].first;
      }
    }
  }
  return minCost;
}

template<typename SplitPolicy>
template<typename TreeType>
size_t MinimalSplitsNumberSweep<SplitPolicy>::SweepLeafNode(
    const size_t axis,
    const TreeType* node,
    typename TreeType::ElemType& axisCut)
{
  // Split along the median.
  axisCut = (node->Bound()[axis].Lo() + node->Bound()[axis].Hi()) * 0.5;

  if (node->Bound()[axis].Lo() == axisCut)
    return SIZE_MAX;

  return 0;
}


} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_IMPL_HPP


