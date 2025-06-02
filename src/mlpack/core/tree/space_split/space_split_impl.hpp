/**
 * @file core/tree/space_split/space_split_impl.hpp
 * @author Marcos Pividori
 *
 * Implementation of SpaceSplit, to create a projection vector based on a given
 * set of points.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_IMPL_HPP

#include "space_split.hpp"
#include <mlpack/core/math/random.hpp>

namespace mlpack {

template<typename DistanceType, typename MatType>
bool SpaceSplit<DistanceType, MatType>::GetProjVector(
    const HRectBound<DistanceType, typename MatType::elem_type>& bound,
    const MatType& data,
    const arma::Col<size_t>& /* points */,
    AxisParallelProjVector& projVector,
    typename MatType::elem_type& midValue)
{
  using ElemType = typename MatType::elem_type;

  // Get the dimension that has the maximum width.
  size_t splitDim = data.n_rows; // Indicate invalid.
  ElemType maxWidth = -1;

  for (size_t d = 0; d < data.n_rows; d++)
  {
    const ElemType width = bound[d].Width();

    if (width > maxWidth)
    {
      maxWidth = width;
      splitDim = d;
    }
  }

  if (maxWidth <= 0) // All these points are the same.
    return false;

  projVector = AxisParallelProjVector(splitDim);

  midValue = bound[splitDim].Mid();

  return true;
}

template<typename DistanceType, typename MatType>
template<typename BoundType>
bool SpaceSplit<DistanceType, MatType>::GetProjVector(
    const BoundType& /* bound */,
    const MatType& data,
    const arma::Col<size_t>& points,
    ProjVector<MatType>& projVector,
    typename MatType::elem_type& midValue)
{
  using ElemType = typename MatType::elem_type;
  using VecType = typename GetColType<MatType>::type;

  DistanceType distance;

  // Efficiently estimate the farthest pair of points in the given set.
  size_t fst = points[RandInt(points.n_elem)];
  size_t snd = points[0];
  ElemType max = distance.Evaluate(data.col(fst), data.col(snd));

  for (size_t i = 1; i < points.n_elem; ++i)
  {
    ElemType dist = distance.Evaluate(data.col(fst), data.col(points[i]));
    if (dist > max)
    {
      max = dist;
      snd = points[i];
    }
  }

  std::swap(fst, snd);

  for (size_t i = 0; i < points.n_elem; ++i)
  {
    ElemType dist = distance.Evaluate(data.col(fst), data.col(points[i]));
    if (dist > max)
    {
      max = dist;
      snd = points[i];
    }
  }

  if (max == 0) // All these points are the same.
    return false;

  // Calculate the normalized projection vector.
  projVector = ProjVector<MatType>(data.col(snd) - data.col(fst));

  VecType midPoint = (data.col(snd) + data.col(fst)) / 2;

  midValue = projVector.Project(midPoint);

  return true;
}

} // namespace mlpack

#endif
