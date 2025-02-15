/**
 * @file core/tree/hrectbound_impl.hpp
 *
 * Implementation of hyper-rectangle bound policy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
#define MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP

#include <math.h>

// In case it has not been included yet.
#include "hrectbound.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

/**
 * Empty constructor.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>::HRectBound() :
    dim(0),
    bounds(NULL),
    minWidth(0)
{ /* Nothing to do. */ }

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>::HRectBound(const size_t dimension) :
    dim(dimension),
    bounds(new RangeType<ElemType>[dim]),
    minWidth(0)
{ /* Nothing to do. */ }

/**
 * Copy constructor necessary to prevent memory leaks.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>::HRectBound(
    const HRectBound<DistanceType, ElemType>& other) :
    dim(other.Dim()),
    bounds(new RangeType<ElemType>[dim]),
    minWidth(other.MinWidth())
{
  // Copy other bounds over.
  for (size_t i = 0; i < dim; ++i)
    bounds[i] = other[i];
}

/**
 * Same as the copy constructor.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<
    DistanceType,
    ElemType>& HRectBound<DistanceType,
    ElemType>::operator=(const HRectBound<DistanceType, ElemType>& other)
{
  if (this == &other)
    return *this;

  if (dim != other.Dim())
  {
    // Reallocation is necessary.
    if (bounds)
      delete[] bounds;

    dim = other.Dim();
    bounds = new RangeType<ElemType>[dim];
  }

  // Now copy each of the bound values.
  for (size_t i = 0; i < dim; ++i)
    bounds[i] = other[i];

  minWidth = other.MinWidth();

  return *this;
}

/**
 * Move constructor: take possession of another bound's information.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>::HRectBound(
    HRectBound<DistanceType, ElemType>&& other) :
    dim(other.dim),
    bounds(other.bounds),
    minWidth(other.minWidth)
{
  // Fix the other bound.
  other.dim = 0;
  other.bounds = NULL;
  other.minWidth = 0.0;
}

/**
 * Move assignment operator.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>&
HRectBound<DistanceType, ElemType>::operator=(
    HRectBound<DistanceType, ElemType>&& other)
{
  if (this != &other)
  {
    bounds = other.bounds;
    minWidth = other.minWidth;
    dim = other.dim;
    other.dim = 0;
    other.bounds = nullptr;
    other.minWidth = 0.0;
  }
  return *this;
}

/**
 * Destructor: clean up memory.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>::~HRectBound()
{
  if (bounds)
    delete[] bounds;
}

/**
 * Resets all dimensions to the empty set.
 */
template<typename DistanceType, typename ElemType>
inline void HRectBound<DistanceType, ElemType>::Clear()
{
  for (size_t i = 0; i < dim; ++i)
    bounds[i] = RangeType<ElemType>();
  minWidth = 0;
}

/***
 * Calculates the centroid of the range, placing it into the given vector.
 *
 * @param centroid Vector which the centroid will be written to.
 */
template<typename DistanceType, typename ElemType>
inline void HRectBound<DistanceType, ElemType>::Center(
    arma::Col<ElemType>& center) const
{
  // Set size correctly if necessary.
  if (!(center.n_elem == dim))
    center.set_size(dim);

  for (size_t i = 0; i < dim; ++i)
    center(i) = bounds[i].Mid();
}

/**
 * Recompute the minimum width of the bound.
 */
template<typename DistanceType, typename ElemType>
inline void HRectBound<DistanceType, ElemType>::RecomputeMinWidth()
{
  minWidth = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < dim; ++i)
    minWidth = std::min(minWidth, bounds[i].Width());
}

/**
 * Calculate the volume of the hyperrectangle.
 *
 * @return Volume of the hyperrectangle.
 */
template<typename DistanceType, typename ElemType>
inline ElemType HRectBound<DistanceType, ElemType>::Volume() const
{
  ElemType volume = 1.0;
  for (size_t i = 0; i < dim; ++i)
  {
    if (bounds[i].Lo() >= bounds[i].Hi())
      return 0;

    volume *= (bounds[i].Hi() - bounds[i].Lo());
  }

  return volume;
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename DistanceType, typename ElemType>
template<typename VecType>
inline ElemType HRectBound<DistanceType, ElemType>::MinDistance(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>* /* junk */) const
{
  Log::Assert(point.n_elem == dim);

  ElemType sum = 0;

  ElemType lower, higher;
  for (size_t d = 0; d < dim; d++)
  {
    lower = bounds[d].Lo() - point[d];
    higher = point[d] - bounds[d].Hi();

    // Since only one of 'lower' or 'higher' is negative, if we add each's
    // absolute value to itself and then sum those two, our result is the
    // nonnegative half of the equation times two; then we raise to power Power.
    if (DistanceType::Power == 1)
      sum += (lower + std::fabs(lower)) + (higher + std::fabs(higher));
    else if (DistanceType::Power == 2)
    {
      ElemType dist = (lower + std::fabs(lower)) + (higher + std::fabs(higher));
      sum += dist * dist;
    }
    else
    {
      sum += std::pow((lower + std::fabs(lower)) + (higher + std::fabs(higher)),
          (ElemType) DistanceType::Power);
    }
  }

  // Now take the Power'th root (but make sure our result is squared if it needs
  // to be); then cancel out the constant of 2 (which may have been squared now)
  // that was introduced earlier.  The compiler should optimize out the if
  // statement entirely.
  if (DistanceType::Power == 1)
    return sum * 0.5;
  else if (DistanceType::Power == 2)
  {
    if (DistanceType::TakeRoot)
      return (ElemType) std::sqrt(sum) * 0.5;
    else
      return sum * 0.25;
  }
  else
  {
    if (DistanceType::TakeRoot)
      return (ElemType) std::pow((double) sum,
          1.0 / (double) DistanceType::Power) / 2.0;
    else
      return sum / std::pow(2.0, DistanceType::Power);
  }
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename DistanceType, typename ElemType>
ElemType HRectBound<DistanceType, ElemType>::MinDistance(
    const HRectBound& other) const
{
  Log::Assert(dim == other.dim);

  ElemType sum = 0;
  const RangeType<ElemType>* mbound = bounds;
  const RangeType<ElemType>* obound = other.bounds;

  ElemType lower, higher;
  for (size_t d = 0; d < dim; d++)
  {
    lower = obound->Lo() - mbound->Hi();
    higher = mbound->Lo() - obound->Hi();
    // We invoke the following:
    //   x + fabs(x) = max(x * 2, 0)
    //   (x * 2)^2 / 4 = x^2

    // The compiler should optimize out this if statement entirely.
    if (DistanceType::Power == 1)
      sum += (lower + std::fabs(lower)) + (higher + std::fabs(higher));
    else if (DistanceType::Power == 2)
    {
      ElemType dist = (lower + std::fabs(lower)) + (higher + std::fabs(higher));
      sum += dist * dist;
    }
    else
    {
      sum += std::pow((lower + std::fabs(lower)) + (higher + std::fabs(higher)),
          (ElemType) DistanceType::Power);
    }

    // Move bound pointers.
    mbound++;
    obound++;
  }

  // The compiler should optimize out this if statement entirely.
  if (DistanceType::Power == 1)
    return sum * 0.5;
  else if (DistanceType::Power == 2)
  {
    if (DistanceType::TakeRoot)
      return (ElemType) std::sqrt(sum) * 0.5;
    else
      return sum * 0.25;
  }
  else
  {
    if (DistanceType::TakeRoot)
      return (ElemType) std::pow((double) sum,
          1.0 / (double) DistanceType::Power) / 2.0;
    else
      return sum / std::pow(2.0, DistanceType::Power);
  }
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<typename DistanceType, typename ElemType>
template<typename VecType>
inline ElemType HRectBound<DistanceType, ElemType>::MaxDistance(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>* /* junk */) const
{
  ElemType sum = 0;

  Log::Assert(point.n_elem == dim);

  for (size_t d = 0; d < dim; d++)
  {
    ElemType v = std::max(fabs(point[d] - bounds[d].Lo()),
        fabs(bounds[d].Hi() - point[d]));

    // The compiler should optimize out this if statement entirely.
    if (DistanceType::Power == 1)
      sum += v; // v is non-negative.
    else if (DistanceType::Power == 2)
      sum += v * v;
    else
      sum += std::pow(v, (ElemType) DistanceType::Power);
  }

  // The compiler should optimize out this if statement entirely.
  if (DistanceType::TakeRoot)
  {
    if (DistanceType::Power == 1)
      return sum;
    else if (DistanceType::Power == 2)
      return (ElemType) std::sqrt(sum);
    else
      return (ElemType) std::pow((double) sum, 1.0 /
          (double) DistanceType::Power);
  }
  else
    return sum;
}

/**
 * Computes maximum distance.
 */
template<typename DistanceType, typename ElemType>
inline ElemType HRectBound<DistanceType, ElemType>::MaxDistance(
    const HRectBound& other)
    const
{
  ElemType sum = 0;

  Log::Assert(dim == other.dim);

  ElemType v;
  for (size_t d = 0; d < dim; d++)
  {
    v = std::max(fabs(other.bounds[d].Hi() - bounds[d].Lo()),
        fabs(bounds[d].Hi() - other.bounds[d].Lo()));

    // The compiler should optimize out this if statement entirely.
    if (DistanceType::Power == 1)
      sum += v; // v is non-negative.
    else if (DistanceType::Power == 2)
      sum += v * v;
    else
      sum += std::pow(v, (ElemType) DistanceType::Power);
  }

  // The compiler should optimize out this if statement entirely.
  if (DistanceType::TakeRoot)
  {
    if (DistanceType::Power == 1)
      return sum;
    else if (DistanceType::Power == 2)
      return (ElemType) std::sqrt(sum);
    else
      return (ElemType) std::pow((double) sum, 1.0 /
          (double) DistanceType::Power);
  }
  else
    return sum;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<typename DistanceType, typename ElemType>
inline RangeType<ElemType>
HRectBound<DistanceType, ElemType>::RangeDistance(
    const HRectBound& other) const
{
  ElemType loSum = 0;
  ElemType hiSum = 0;

  Log::Assert(dim == other.dim);

  ElemType v1, v2, vLo, vHi;
  for (size_t d = 0; d < dim; d++)
  {
    v1 = other.bounds[d].Lo() - bounds[d].Hi();
    v2 = bounds[d].Lo() - other.bounds[d].Hi();
    // One of v1 or v2 is negative.
    if (v1 >= v2)
    {
      vHi = -v2; // Make it nonnegative.
      vLo = (v1 > 0) ? v1 : 0; // Force to be 0 if negative.
    }
    else
    {
      vHi = -v1; // Make it nonnegative.
      vLo = (v2 > 0) ? v2 : 0; // Force to be 0 if negative.
    }

    // The compiler should optimize out this if statement entirely.
    if (DistanceType::Power == 1)
    {
      loSum += vLo; // vLo is non-negative.
      hiSum += vHi; // vHi is non-negative.
    }
    else if (DistanceType::Power == 2)
    {
      loSum += vLo * vLo;
      hiSum += vHi * vHi;
    }
    else
    {
      loSum += std::pow(vLo, (ElemType) DistanceType::Power);
      hiSum += std::pow(vHi, (ElemType) DistanceType::Power);
    }
  }

  if (DistanceType::TakeRoot)
  {
    if (DistanceType::Power == 1)
      return RangeType<ElemType>(loSum, hiSum);
    else if (DistanceType::Power == 2)
      return RangeType<ElemType>((ElemType) std::sqrt(loSum),
                                       (ElemType) std::sqrt(hiSum));
    else
    {
      return RangeType<ElemType>(
          (ElemType) std::pow((double) loSum,
              1.0 / (double) DistanceType::Power),
          (ElemType) std::pow((double) hiSum,
              1.0 / (double) DistanceType::Power));
    }
  }
  else
  {
    return RangeType<ElemType>(loSum, hiSum);
  }
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<typename DistanceType, typename ElemType>
template<typename VecType>
inline RangeType<ElemType>
HRectBound<DistanceType, ElemType>::RangeDistance(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>* /* junk */) const
{
  ElemType loSum = 0;
  ElemType hiSum = 0;

  Log::Assert(point.n_elem == dim);

  ElemType v1, v2, vLo, vHi;
  for (size_t d = 0; d < dim; d++)
  {
    v1 = bounds[d].Lo() - point[d]; // Negative if point[d] > lo.
    v2 = point[d] - bounds[d].Hi(); // Negative if point[d] < hi.
    // One of v1 or v2 (or both) is negative.
    if (v1 >= 0) // point[d] <= bounds_[d].Lo().
    {
      vHi = -v2; // v2 will be larger but must be negated.
      vLo = v1;
    }
    else // point[d] is between lo and hi, or greater than hi.
    {
      if (v2 >= 0)
      {
        vHi = -v1; // v1 will be larger, but must be negated.
        vLo = v2;
      }
      else
      {
        vHi = -std::min(v1, v2); // Both are negative, but we need the larger.
        vLo = 0;
      }
    }

    // The compiler should optimize out this if statement entirely.
    if (DistanceType::Power == 1)
    {
      loSum += vLo; // vLo is non-negative.
      hiSum += vHi; // vHi is non-negative.
    }
    else if (DistanceType::Power == 2)
    {
      loSum += vLo * vLo;
      hiSum += vHi * vHi;
    }
    else
    {
      loSum += std::pow(vLo, (ElemType) DistanceType::Power);
      hiSum += std::pow(vHi, (ElemType) DistanceType::Power);
    }
  }

  if (DistanceType::TakeRoot)
  {
    if (DistanceType::Power == 1)
      return RangeType<ElemType>(loSum, hiSum);
    else if (DistanceType::Power == 2)
      return RangeType<ElemType>((ElemType) std::sqrt(loSum),
                                 (ElemType) std::sqrt(hiSum));
    else
    {
      return RangeType<ElemType>(
          (ElemType) std::pow((double) loSum,
              1.0 / (double) DistanceType::Power),
          (ElemType) std::pow((double) hiSum,
              1.0 / (double) DistanceType::Power));
    }
  }
  else
    return RangeType<ElemType>(loSum, hiSum);
}

/**
 * Expands this region to include a new point.
 */
template<typename DistanceType, typename ElemType>
template<typename MatType>
inline HRectBound<DistanceType, ElemType>&
HRectBound<DistanceType, ElemType>::operator|=(const MatType& data)
{
  if (dim == 0)
  {
    delete[] bounds;
    dim = data.n_rows;
    bounds = new RangeType<ElemType>[dim];
  }

  Log::Assert(data.n_rows == dim);

  arma::Col<ElemType> mins(min(data, 1));
  arma::Col<ElemType> maxs(max(data, 1));

  minWidth = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < dim; ++i)
  {
    bounds[i] |= RangeType<ElemType>(mins[i], maxs[i]);
    const ElemType width = bounds[i].Width();
    if (width < minWidth)
      minWidth = width;
  }

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>&
HRectBound<DistanceType, ElemType>::operator|=(const HRectBound& other)
{
  if (dim == 0)
  {
    delete[] bounds;
    dim = other.dim;
    bounds = new RangeType<ElemType>[dim];
  }

  Log::Assert(other.dim == dim);

  minWidth = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < dim; ++i)
  {
    bounds[i] |= other.bounds[i];
    const ElemType width = bounds[i].Width();
    if (width < minWidth)
      minWidth = width;
  }

  return *this;
}

/**
 * Determines if a point is within this bound.
 */
template<typename DistanceType, typename ElemType>
template<typename VecType>
inline bool HRectBound<DistanceType, ElemType>::Contains(
    const VecType& point) const
{
  for (size_t i = 0; i < point.n_elem; ++i)
  {
    if (!bounds[i].Contains(point(i)))
      return false;
  }

  return true;
}

/**
 * Determines if this bound partially contains a bound.
 */
template<typename DistanceType, typename ElemType>
inline bool HRectBound<DistanceType, ElemType>::Contains(
    const HRectBound& bound) const
{
  for (size_t i = 0; i < dim; ++i)
  {
    const RangeType<ElemType>& r_a = bounds[i];
    const RangeType<ElemType>& r_b = bound.bounds[i];

    // If a does not overlap b at all.
    if (r_a.Hi() <= r_b.Lo() || r_a.Lo() >= r_b.Hi())
      return false;
  }

  return true;
}

/**
 * Returns the intersection of this bound and another.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>
HRectBound<DistanceType, ElemType>::operator&(const HRectBound& bound) const
{
  HRectBound<DistanceType, ElemType> result(dim);

  for (size_t k = 0; k < dim; ++k)
  {
    result[k].Lo() = std::max(bounds[k].Lo(), bound.bounds[k].Lo());
    result[k].Hi() = std::min(bounds[k].Hi(), bound.bounds[k].Hi());
  }
  return result;
}

/**
 * Intersects this bound with another.
 */
template<typename DistanceType, typename ElemType>
inline HRectBound<DistanceType, ElemType>&
HRectBound<DistanceType, ElemType>::operator&=(const HRectBound& bound)
{
  for (size_t k = 0; k < dim; ++k)
  {
    bounds[k].Lo() = std::max(bounds[k].Lo(), bound.bounds[k].Lo());
    bounds[k].Hi() = std::min(bounds[k].Hi(), bound.bounds[k].Hi());
  }
  return *this;
}

/**
 * Returns the volume of overlap of this bound and another.
 */
template<typename DistanceType, typename ElemType>
inline ElemType HRectBound<DistanceType, ElemType>::Overlap(
    const HRectBound& bound) const
{
  ElemType volume = 1.0;

  for (size_t k = 0; k < dim; ++k)
  {
    ElemType lo = std::max(bounds[k].Lo(), bound.bounds[k].Lo());
    ElemType hi = std::min(bounds[k].Hi(), bound.bounds[k].Hi());

    if (hi <= lo)
      return 0;

    volume *= hi - lo;
  }
  return volume;
}

/**
 * Returns the diameter of the hyperrectangle (that is, the longest diagonal).
 */
template<typename DistanceType, typename ElemType>
inline ElemType HRectBound<DistanceType, ElemType>::Diameter() const
{
  ElemType d = 0;
  for (size_t i = 0; i < dim; ++i)
    d += std::pow(bounds[i].Hi() - bounds[i].Lo(),
        (ElemType) DistanceType::Power);

  if (DistanceType::TakeRoot)
    return (ElemType) std::pow((double) d, 1.0 / (double) DistanceType::Power);
  else
    return d;
}

//! Serialize the bound object.
template<typename DistanceType, typename ElemType>
template<typename Archive>
void HRectBound<DistanceType, ElemType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  // We can't serialize a raw array directly, so wrap it.
  ar(CEREAL_POINTER_ARRAY(bounds, dim));
  ar(CEREAL_NVP(minWidth));
  ar(CEREAL_NVP(distance));
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
