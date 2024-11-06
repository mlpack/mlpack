/**
 * @file core/tree/rectangle_tree/minimal_coverage_sweep_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the MinimalCoverageSweep class, a class that finds a
 * partition of a node along an axis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_IMPL_HPP

#include "minimal_coverage_sweep.hpp"

namespace mlpack {

template<typename SplitPolicy>
template<typename TreeType>
typename TreeType::ElemType MinimalCoverageSweep<SplitPolicy>::
SweepNonLeafNode(const size_t axis,
                 const TreeType* node,
                 typename TreeType::ElemType& axisCut)
{
  using ElemType = typename TreeType::ElemType;
  using BoundType = HRectBound<EuclideanDistance, ElemType>;

  std::vector<std::pair<ElemType, size_t>> sorted(node->NumChildren());

  for (size_t i = 0; i < node->NumChildren(); ++i)
  {
    sorted[i].first = SplitPolicy::Bound(node->Child(i))[axis].Hi();
    sorted[i].second = i;
  }
  // Sort high bounds of children.
  std::sort(sorted.begin(), sorted.end(),
      [] (const std::pair<ElemType, size_t>& s1,
          const std::pair<ElemType, size_t>& s2)
      {
        return s1.first < s2.first;
      });

  size_t splitPointer = node->NumChildren() / 2;

  axisCut = sorted[splitPointer - 1].first;

  // Check if the midpoint split is suitable.
  if (!CheckNonLeafSweep(node, axis, axisCut))
  {
    // Find any suitable partition if the default partition is not acceptable.
    for (splitPointer = 1; splitPointer < sorted.size(); splitPointer++)
    {
      axisCut = sorted[splitPointer - 1].first;
      if (CheckNonLeafSweep(node, axis, axisCut))
        break;
    }

    if (splitPointer == node->NumChildren())
      return std::numeric_limits<ElemType>::max();
  }

  BoundType bound1(node->Bound().Dim());
  BoundType bound2(node->Bound().Dim());

  // Find bounds of two resulting nodes.
  for (size_t i = 0; i < splitPointer; ++i)
    bound1 |= node->Child(sorted[i].second).Bound();

  for (size_t i = splitPointer; i < node->NumChildren(); ++i)
    bound2 |= node->Child(sorted[i].second).Bound();


  // Evaluate the cost of the split i.e. calculate the total coverage
  // of two resulting nodes.

  ElemType area1 = bound1.Volume();
  ElemType area2 = bound2.Volume();

  return area1 + area2;
}

template<typename SplitPolicy>
template<typename TreeType>
typename TreeType::ElemType MinimalCoverageSweep<SplitPolicy>::
SweepLeafNode(const size_t axis,
              const TreeType* node,
              typename TreeType::ElemType& axisCut)
{
  using ElemType = typename TreeType::ElemType;
  using BoundType = HRectBound<EuclideanDistance, ElemType>;

  std::vector<std::pair<ElemType, size_t>> sorted(node->Count());

  sorted.resize(node->Count());

  for (size_t i = 0; i < node->NumPoints(); ++i)
  {
    sorted[i].first = node->Dataset().col(node->Point(i))[axis];
    sorted[i].second = i;
  }

  // Sort high bounds of children.
  std::sort(sorted.begin(), sorted.end(),
      [] (const std::pair<ElemType, size_t>& s1,
          const std::pair<ElemType, size_t>& s2)
      {
        return s1.first < s2.first;
      });

  size_t splitPointer = node->Count() / 2;

  axisCut = sorted[splitPointer - 1].first;

  // Check if the partition is suitable.
  if (!CheckLeafSweep(node, axis, axisCut))
    return std::numeric_limits<ElemType>::max();

  BoundType bound1(node->Bound().Dim());
  BoundType bound2(node->Bound().Dim());

  // Find bounds of two resulting nodes.
  for (size_t i = 0; i < splitPointer; ++i)
    bound1 |= node->Dataset().col(node->Point(sorted[i].second));

  for (size_t i = splitPointer; i < node->NumChildren(); ++i)
    bound2 |= node->Dataset().col(node->Point(sorted[i].second));

  // Evaluate the cost of the split i.e. calculate the total coverage
  // of two resulting nodes.

  return bound1.Volume() + bound2.Volume();
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
  for (size_t i = 0; i < node->NumChildren(); ++i)
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
  for (size_t i = 0; i < node->NumPoints(); ++i)
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

} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_MINIMAL_COVERAGE_SWEEP_IMPL_HPP

