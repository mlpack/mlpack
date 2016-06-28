/**
 * @file minimal_splits_number_sweep_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the MinimalSplitsNumberSweep class, a class that finds a
 * partition of a node along an axis.
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

  std::vector<SortStruct<ElemType>> sorted(node->NumChildren());

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    sorted[i].d = SplitPolicy::Bound(node->Children()[i])[axis].Hi();
    sorted[i].n = i;
  }

  // Sort candidates in order to check balancing.
  std::sort(sorted.begin(), sorted.end(), StructComp<ElemType>);

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
      TreeType* child = node->Children()[j];
      int policy = SplitPolicy::GetSplitPolicy(child, axis, sorted[i].d);
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
      size_t cost = numSplits * (std::abs(sorted.size() / 2 - i));
      if (cost < minCost)
      {
        minCost = cost;
        axisCut = sorted[i].d;
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


