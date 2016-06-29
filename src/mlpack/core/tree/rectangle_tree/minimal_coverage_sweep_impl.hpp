/**
 * @file minimal_coverage_sweep_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the MinimalCoverageSweep class, a class that finds a
 * partition of a node along an axis.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_IMPL_HPP

#include "minimal_coverage_sweep.hpp"

namespace mlpack {
namespace tree {

template<typename SplitPolicy>
template<typename TreeType>
typename TreeType::ElemType MinimalCoverageSweep<SplitPolicy>::
SweepNonLeafNode(const size_t axis,
                 const TreeType* node,
                 typename TreeType::ElemType& axisCut)
{
  typedef typename TreeType::ElemType ElemType;

  std::vector<SortStruct<ElemType>> sorted(node->NumChildren());

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    sorted[i].d = SplitPolicy::Bound(node->Child(i))[axis].Hi();
    sorted[i].n = i;
  }
  // Sort high bounds of children.
  std::sort(sorted.begin(), sorted.end(), StructComp<ElemType>);

  size_t splitPointer = fillFactor * node->NumChildren();

  axisCut = sorted[splitPointer - 1].d;

  // Check if the partition is suitable.
  if (!CheckNonLeafSweep(node, axis, axisCut))
  {
    for (splitPointer = 1; splitPointer < sorted.size(); splitPointer++)
    {
      axisCut = sorted[splitPointer - 1].d;
      if (CheckNonLeafSweep(node, axis, axisCut))
        break;
    }

    if (splitPointer == node->NumChildren())
      return std::numeric_limits<ElemType>::max();
  }

  std::vector<ElemType> lowerBound1(node->Bound().Dim());
  std::vector<ElemType> highBound1(node->Bound().Dim());
  std::vector<ElemType> lowerBound2(node->Bound().Dim());
  std::vector<ElemType> highBound2(node->Bound().Dim());

  // Find lower and high bounds of two resulting nodes.
  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    lowerBound1[k] = node->Child(sorted[0].n).Bound()[k].Lo();
    highBound1[k] = node->Child(sorted[0].n).Bound()[k].Hi();

    for (size_t i = 1; i < splitPointer; i++)
    {
      if (node->Child(sorted[i].n).Bound()[k].Lo() < lowerBound1[k])
        lowerBound1[k] = node->Child(sorted[i].n).Bound()[k].Lo();
      if (node->Child(sorted[i].n).Bound()[k].Hi() > highBound1[k])
        highBound1[k] = node->Child(sorted[i].n).Bound()[k].Hi();
    }

    lowerBound2[k] = node->Child(sorted[splitPointer].n).Bound()[k].Lo();
    highBound2[k] = node->Child(sorted[splitPointer].n).Bound()[k].Hi();

    for (size_t i = splitPointer + 1; i < node->NumChildren(); i++)
    {
      if (node->Child(sorted[i].n).Bound()[k].Lo() < lowerBound2[k])
        lowerBound2[k] = node->Child(sorted[i].n).Bound()[k].Lo();
      if (node->Child(sorted[i].n).Bound()[k].Hi() > highBound2[k])
        highBound2[k] = node->Child(sorted[i].n).Bound()[k].Hi();
    }
  }

  // Evaluate the cost of the split i.e. calculate the total coverage
  // of two resulting nodes.

  ElemType area1 = 1.0, area2 = 1.0;
  ElemType overlappedArea = 1.0;

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    if (lowerBound1[k] >= highBound1[k])
    {
      overlappedArea *= 0;
      area1 *= 0;
    }
    else
      area1 *= highBound1[k] - lowerBound1[k];

    if (lowerBound2[k] >= highBound2[k])
    {
      overlappedArea *= 0;
      area1 *= 0;
    }
    else
      area2 *= highBound2[k] - lowerBound2[k];

    if (lowerBound1[k] < highBound1[k] && lowerBound2[k] < highBound2[k])
    {
      if (lowerBound1[k] > highBound2[k] || lowerBound2[k] > highBound2[k])
        overlappedArea *= 0;
      else
        overlappedArea *= std::min(highBound1[k], highBound2[k]) -
            std::max(lowerBound1[k], lowerBound2[k]);
    }
  }

  return area1 + area2 - overlappedArea;
}

template<typename SplitPolicy>
template<typename TreeType>
typename TreeType::ElemType MinimalCoverageSweep<SplitPolicy>::
SweepLeafNode(const size_t axis,
              const TreeType* node,
              typename TreeType::ElemType& axisCut)
{
  typedef typename TreeType::ElemType ElemType;

  std::vector<SortStruct<ElemType>> sorted(node->Count());

  sorted.resize(node->Count());

  for (size_t i = 0; i < node->NumPoints(); i++)
  {
    sorted[i].d = node->Dataset().col(node->Point(i))[axis];
    sorted[i].n = i;
  }

  // Sort high bounds of children.
  std::sort(sorted.begin(), sorted.end(), StructComp<ElemType>);

  size_t splitPointer = fillFactor * node->Count();

  axisCut = sorted[splitPointer - 1].d;

  // Check if the partition is suitable.
  if (!CheckLeafSweep(node, axis, axisCut))
    return std::numeric_limits<ElemType>::max();

  std::vector<ElemType> lowerBound1(node->Bound().Dim());
  std::vector<ElemType> highBound1(node->Bound().Dim());
  std::vector<ElemType> lowerBound2(node->Bound().Dim());
  std::vector<ElemType> highBound2(node->Bound().Dim());

  // Find lower and high bounds of two resulting nodes.
  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    lowerBound1[k] = node->Dataset().col(node->Point(sorted[0].n))[k];
    highBound1[k] = node->Dataset().col(node->Point(sorted[0].n))[k];

    for (size_t i = 1; i < splitPointer; i++)
    {
      if (node->Dataset().col(node->Point(sorted[i].n))[k] < lowerBound1[k])
        lowerBound1[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
      if (node->Dataset().col(node->Point(sorted[i].n))[k] > highBound1[k])
        highBound1[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
    }

    lowerBound2[k] = node->Dataset().col(
        node->Point(sorted[splitPointer].n))[k];
    highBound2[k] = node->Dataset().col(node->Point(sorted[splitPointer].n))[k];

    for (size_t i = splitPointer + 1; i < node->NumChildren(); i++)
    {
      if (node->Dataset().col(node->Point(sorted[i].n))[k] < lowerBound2[k])
        lowerBound2[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
      if (node->Dataset().col(node->Point(sorted[i].n))[k] > highBound2[k])
        highBound2[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
    }
  }

  // Evaluate the cost of the split i.e. calculate the total coverage
  // of two resulting nodes.

  ElemType area1 = 1.0, area2 = 1.0;
  ElemType overlappedArea = 1.0;

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    area1 *= highBound1[k] - lowerBound1[k];
    area2 *= highBound2[k] - lowerBound2[k];
  }

  return area1 + area2 - overlappedArea;
}

template<typename SplitPolicy>
template<typename TreeType, typename ElemType>
bool MinimalCoverageSweep<SplitPolicy>::
CheckNonLeafSweep(const TreeType* node,
                  const size_t cutAxis,
                  const ElemType cut)
{
  size_t numTreeOneChildren = 0;
  size_t numTreeTwoChildren = 0;

  // Calculate the number of children in the resulting nodes.
  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    const TreeType& child = node->Child(i);
    int policy = SplitPolicy::GetSplitPolicy(child, cutAxis, cut);
    if (policy == SplitPolicy::AssignToFirstTree)
      numTreeOneChildren++;
    else if (policy == SplitPolicy::AssignToSecondTree)
      numTreeTwoChildren++;
    else
    {
      // The split is required.
      numTreeOneChildren++;
      numTreeTwoChildren++;
    }
  }

  if (numTreeOneChildren <= node->MaxNumChildren() && numTreeOneChildren > 0 &&
      numTreeTwoChildren <= node->MaxNumChildren() && numTreeTwoChildren > 0)
    return true;
  return false;
}

template<typename SplitPolicy>
template<typename TreeType, typename ElemType>
bool MinimalCoverageSweep<SplitPolicy>::
CheckLeafSweep(const TreeType* node,
               const size_t cutAxis,
               const ElemType cut)
{
  size_t numTreeOnePoints = 0;
  size_t numTreeTwoPoints = 0;

  // Calculate the number of points in the resulting nodes.
  for (size_t i = 0; i < node->NumPoints(); i++)
  {
    if (node->Dataset().col(node->Point(i))[cutAxis] <= cut)
      numTreeOnePoints++;
    else
      numTreeTwoPoints++;
  }

  if (numTreeOnePoints <= node->MaxLeafSize() && numTreeOnePoints > 0 &&
      numTreeTwoPoints <= node->MaxLeafSize() && numTreeTwoPoints > 0)
    return true;
  return false;
}

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_IMPL_HPP

