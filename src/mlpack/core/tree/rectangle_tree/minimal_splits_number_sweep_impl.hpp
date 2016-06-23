/**
 * @file minimal_splits_number_sweep_impl.hpp
 * @author Mikhail Lozhnikov
 *
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_IMPL_HPP

#include "minimal_splits_number_sweep.hpp"

namespace mlpack {
namespace tree {

template<typename SplitPolicy>
template<typename TreeType>
size_t MinimalSplitsNumberSweep<SplitPolicy>::
SweepNonLeafNode(size_t axis, const TreeType* node,
    typename TreeType::ElemType& axisCut)
{
  typedef typename TreeType::ElemType ElemType;

  std::vector<SortStruct<ElemType>> sorted(node->NumChildren());

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    sorted[i].d = SplitPolicy::Bound(node->Children()[i])[axis].Hi();
    sorted[i].n = i;
  }
  std::sort(sorted.begin(), sorted.end(), StructComp<ElemType>);

  size_t minCost = SIZE_MAX;

  for (size_t i = 0; i < sorted.size(); i++)
  {
    size_t numTreeOneChildren = 0;
    size_t numTreeTwoChildren = 0;
    size_t numSplits = 0;

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

    if (numTreeOneChildren <= node->MaxNumChildren() && numTreeOneChildren > 0 &&
        numTreeTwoChildren <= node->MaxNumChildren() && numTreeTwoChildren > 0)
    {
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
size_t MinimalSplitsNumberSweep<SplitPolicy>::
SweepLeafNode(size_t axis, const TreeType* node,
    typename TreeType::ElemType& axisCut)
{
  axisCut = (node->Bound()[axis].Lo() + node->Bound()[axis].Hi()) * 0.5;

  if (node->Bound()[axis].Lo() == axisCut)
    return SIZE_MAX;

  return 0;
}


} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_SPLITS_NUMBER_SWEEP_IMPL_HPP


