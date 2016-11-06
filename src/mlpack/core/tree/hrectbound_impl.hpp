/**
 * @file hrectbound_impl.hpp
 *
 * Implementation of hyper-rectangle bound policy class.
 * Template parameter Power is the metric to use; use 2 for Euclidean (L2).
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

namespace mlpack {
namespace bound {

/**
 * Empty constructor.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>::HRectBound() :
    dim(0),
    bounds(NULL),
    minWidth(0)
{ /* Nothing to do. */ }

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>::HRectBound(const size_t dimension) :
    dim(dimension),
    bounds(new math::RangeType<ElemType>[dim]),
    minWidth(0)
{ /* Nothing to do. */ }

/**
 * Copy constructor necessary to prevent memory leaks.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>::HRectBound(
    const HRectBound<MetricType, ElemType>& other) :
    dim(other.Dim()),
    bounds(new math::RangeType<ElemType>[dim]),
    minWidth(other.MinWidth())
{
  // Copy other bounds over.
  for (size_t i = 0; i < dim; i++)
    bounds[i] = other[i];
}

/**
 * Same as the copy constructor.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>& HRectBound<MetricType, ElemType>::operator=(
    const HRectBound<MetricType, ElemType>& other)
{
  if (dim != other.Dim())
  {
    // Reallocation is necessary.
    if (bounds)
      delete[] bounds;

    dim = other.Dim();
    bounds = new math::RangeType<ElemType>[dim];
  }

  // Now copy each of the bound values.
  for (size_t i = 0; i < dim; i++)
    bounds[i] = other[i];

  minWidth = other.MinWidth();

  return *this;
}

/**
 * Move constructor: take possession of another bound's information.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>::HRectBound(
    HRectBound<MetricType, ElemType>&& other) :
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
 * Destructor: clean up memory.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>::~HRectBound()
{
  if (bounds)
    delete[] bounds;
}

/**
 * Resets all dimensions to the empty set.
 */
template<typename MetricType, typename ElemType>
inline void HRectBound<MetricType, ElemType>::Clear()
{
  for (size_t i = 0; i < dim; i++)
    bounds[i] = math::RangeType<ElemType>();
  minWidth = 0;
}

/***
 * Calculates the centroid of the range, placing it into the given vector.
 *
 * @param centroid Vector which the centroid will be written to.
 */
template<typename MetricType, typename ElemType>
inline void HRectBound<MetricType, ElemType>::Center(
    arma::Col<ElemType>& center) const
{
  // Set size correctly if necessary.
  if (!(center.n_elem == dim))
    center.set_size(dim);

  for (size_t i = 0; i < dim; i++)
    center(i) = bounds[i].Mid();
}

/**
 * Calculate the volume of the hyperrectangle.
 *
 * @return Volume of the hyperrectangle.
 */
template<typename MetricType, typename ElemType>
inline ElemType HRectBound<MetricType, ElemType>::Volume() const
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
template<typename MetricType, typename ElemType>
template<typename VecType>
inline ElemType HRectBound<MetricType, ElemType>::MinDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType>>* /* junk */) const
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
    if (MetricType::Power == 1)
      sum += (lower + std::fabs(lower)) + (higher + std::fabs(higher));
    else if (MetricType::Power == 2)
    {
      ElemType dist = (lower + std::fabs(lower)) + (higher + std::fabs(higher));
      sum += dist * dist;
    }
    else
    {
      sum += pow((lower + fabs(lower)) + (higher + fabs(higher)),
          (ElemType) MetricType::Power);
    }
  }

  // Now take the Power'th root (but make sure our result is squared if it needs
  // to be); then cancel out the constant of 2 (which may have been squared now)
  // that was introduced earlier.  The compiler should optimize out the if
  // statement entirely.
  if (MetricType::Power == 1)
    return sum * 0.5;
  else if (MetricType::Power == 2)
  {
    if (MetricType::TakeRoot)
      return (ElemType) std::sqrt(sum) * 0.5;
    else
      return sum * 0.25;
  }
  else
  {
    if (MetricType::TakeRoot)
      return (ElemType) pow((double) sum, 1.0 / (double) MetricType::Power) / 2.0;
    else
      return sum / pow(2.0, MetricType::Power);
  }
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename MetricType, typename ElemType>
ElemType HRectBound<MetricType, ElemType>::MinDistance(const HRectBound& other)
    const
{
  Log::Assert(dim == other.dim);

  ElemType sum = 0;
  const math::RangeType<ElemType>* mbound = bounds;
  const math::RangeType<ElemType>* obound = other.bounds;

  ElemType lower, higher;
  for (size_t d = 0; d < dim; d++)
  {
    lower = obound->Lo() - mbound->Hi();
    higher = mbound->Lo() - obound->Hi();
    // We invoke the following:
    //   x + fabs(x) = max(x * 2, 0)
    //   (x * 2)^2 / 4 = x^2

    // The compiler should optimize out this if statement entirely.
    if (MetricType::Power == 1)
      sum += (lower + std::fabs(lower)) + (higher + std::fabs(higher));
    else if (MetricType::Power == 2)
    {
      ElemType dist = (lower + std::fabs(lower)) + (higher + std::fabs(higher));
      sum += dist * dist;
    }
    else
    {
      sum += pow((lower + fabs(lower)) + (higher + fabs(higher)),
          (ElemType) MetricType::Power);
    }

    // Move bound pointers.
    mbound++;
    obound++;
  }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::Power == 1)
    return sum * 0.5;
  else if (MetricType::Power == 2)
  {
    if (MetricType::TakeRoot)
      return (ElemType) std::sqrt(sum) * 0.5;
    else
      return sum * 0.25;
  }
  else
  {
    if (MetricType::TakeRoot)
      return (ElemType) pow((double) sum, 1.0 / (double) MetricType::Power) / 2.0;
    else
      return sum / pow(2.0, MetricType::Power);
  }
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<typename MetricType, typename ElemType>
template<typename VecType>
inline ElemType HRectBound<MetricType, ElemType>::MaxDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >* /* junk */) const
{
  ElemType sum = 0;

  Log::Assert(point.n_elem == dim);

  for (size_t d = 0; d < dim; d++)
  {
    ElemType v = std::max(fabs(point[d] - bounds[d].Lo()),
        fabs(bounds[d].Hi() - point[d]));

    // The compiler should optimize out this if statement entirely.
    if (MetricType::Power == 1)
      sum += v; // v is non-negative.
    else if (MetricType::Power == 2)
      sum += v * v;
    else
      sum += std::pow(v, (ElemType) MetricType::Power);
  }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return sum;
    else if (MetricType::Power == 2)
      return (ElemType) std::sqrt(sum);
    else
      return (ElemType) pow((double) sum, 1.0 / (double) MetricType::Power);
  }
  else
    return sum;
}

/**
 * Computes maximum distance.
 */
template<typename MetricType, typename ElemType>
inline ElemType HRectBound<MetricType, ElemType>::MaxDistance(
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
    if (MetricType::Power == 1)
      sum += v; // v is non-negative.
    else if (MetricType::Power == 2)
      sum += v * v;
    else
      sum += std::pow(v, (ElemType) MetricType::Power);
  }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return sum;
    else if (MetricType::Power == 2)
      return (ElemType) std::sqrt(sum);
    else
      return (ElemType) pow((double) sum, 1.0 / (double) MetricType::Power);
  }
  else
    return sum;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<typename MetricType, typename ElemType>
inline math::RangeType<ElemType>
HRectBound<MetricType, ElemType>::RangeDistance(
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
    if (MetricType::Power == 1)
    {
      loSum += vLo; // vLo is non-negative.
      hiSum += vHi; // vHi is non-negative.
    }
    else if (MetricType::Power == 2)
    {
      loSum += vLo * vLo;
      hiSum += vHi * vHi;
    }
    else
    {
      loSum += std::pow(vLo, (ElemType) MetricType::Power);
      hiSum += std::pow(vHi, (ElemType) MetricType::Power);
    }
  }

  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return math::RangeType<ElemType>(loSum, hiSum);
    else if (MetricType::Power == 2)
      return math::RangeType<ElemType>((ElemType) std::sqrt(loSum),
                                       (ElemType) std::sqrt(hiSum));
    else
    {
      return math::RangeType<ElemType>(
          (ElemType) pow((double) loSum, 1.0 / (double) MetricType::Power),
          (ElemType) pow((double) hiSum, 1.0 / (double) MetricType::Power));
    }
  }
  else
    return math::RangeType<ElemType>(loSum, hiSum);
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<typename MetricType, typename ElemType>
template<typename VecType>
inline math::RangeType<ElemType>
HRectBound<MetricType, ElemType>::RangeDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType>>* /* junk */) const
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
    if (MetricType::Power == 1)
    {
      loSum += vLo; // vLo is non-negative.
      hiSum += vHi; // vHi is non-negative.
    }
    else if (MetricType::Power == 2)
    {
      loSum += vLo * vLo;
      hiSum += vHi * vHi;
    }
    else
    {
      loSum += std::pow(vLo, (ElemType) MetricType::Power);
      hiSum += std::pow(vHi, (ElemType) MetricType::Power);
    }
  }

  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return math::RangeType<ElemType>(loSum, hiSum);
    else if (MetricType::Power == 2)
      return math::RangeType<ElemType>((ElemType) std::sqrt(loSum),
                                       (ElemType) std::sqrt(hiSum));
    else
    {
      return math::RangeType<ElemType>(
          (ElemType) pow((double) loSum, 1.0 / (double) MetricType::Power),
          (ElemType) pow((double) hiSum, 1.0 / (double) MetricType::Power));
    }
  }
  else
    return math::RangeType<ElemType>(loSum, hiSum);
}

/**
 * Expands this region to include a new point.
 */
template<typename MetricType, typename ElemType>
template<typename MatType>
inline HRectBound<MetricType, ElemType>& HRectBound<MetricType, ElemType>::operator|=(
    const MatType& data)
{
  Log::Assert(data.n_rows == dim);

  arma::Col<ElemType> mins(min(data, 1));
  arma::Col<ElemType> maxs(max(data, 1));

  minWidth = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < dim; i++)
  {
    bounds[i] |= math::RangeType<ElemType>(mins[i], maxs[i]);
    const ElemType width = bounds[i].Width();
    if (width < minWidth)
      minWidth = width;
  }

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>& HRectBound<MetricType, ElemType>::operator|=(
    const HRectBound& other)
{
  assert(other.dim == dim);

  minWidth = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < dim; i++)
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
template<typename MetricType, typename ElemType>
template<typename VecType>
inline bool HRectBound<MetricType, ElemType>::Contains(const VecType& point) const
{
  for (size_t i = 0; i < point.n_elem; i++)
  {
    if (!bounds[i].Contains(point(i)))
      return false;
  }

  return true;
}

/**
 * Determines if this bound partially contains a bound.
 */
template<typename MetricType, typename ElemType>
inline bool HRectBound<MetricType, ElemType>::Contains(
    const HRectBound& bound) const
{
  for (size_t i = 0; i < dim; i++)
  {
    const math::RangeType<ElemType>& r_a = bounds[i];
    const math::RangeType<ElemType>& r_b = bound.bounds[i];

    if (r_a.Hi() <= r_b.Lo() || r_a.Lo() >= r_b.Hi()) // If a does not overlap b at all.
      return false;
  }

  return true;
}

/**
 * Returns the intersection of this bound and another.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType> HRectBound<MetricType, ElemType>::
operator&(const HRectBound& bound) const
{
  HRectBound<MetricType, ElemType> result(dim);

  for (size_t k = 0; k < dim; k++)
  {
    result[k].Lo() = std::max(bounds[k].Lo(), bound.bounds[k].Lo());
    result[k].Hi() = std::min(bounds[k].Hi(), bound.bounds[k].Hi());
  }
  return result;
}

/**
 * Intersects this bound with another.
 */
template<typename MetricType, typename ElemType>
inline HRectBound<MetricType, ElemType>& HRectBound<MetricType, ElemType>::
operator&=(const HRectBound& bound)
{
  for (size_t k = 0; k < dim; k++)
  {
    bounds[k].Lo() = std::max(bounds[k].Lo(), bound.bounds[k].Lo());
    bounds[k].Hi() = std::min(bounds[k].Hi(), bound.bounds[k].Hi());
  }
  return *this;
}

/**
 * Returns the volume of overlap of this bound and another.
 */
template<typename MetricType, typename ElemType>
inline ElemType HRectBound<MetricType, ElemType>::Overlap(
    const HRectBound& bound) const
{
  ElemType volume = 1.0;

  for (size_t k = 0; k < dim; k++)
  {
    ElemType lo = std::max(bounds[k].Lo(), bound.bounds[k].Lo());
    ElemType hi = std::min(bounds[k].Hi(), bound.bounds[k].Hi());

    if ( hi <= lo)
      return 0;

    volume *= hi - lo;
  }
  return volume;
}

/**
 * Returns the diameter of the hyperrectangle (that is, the longest diagonal).
 */
template<typename MetricType, typename ElemType>
inline ElemType HRectBound<MetricType, ElemType>::Diameter() const
{
  ElemType d = 0;
  for (size_t i = 0; i < dim; ++i)
    d += std::pow(bounds[i].Hi() - bounds[i].Lo(),
        (ElemType) MetricType::Power);

  if (MetricType::TakeRoot)
    return (ElemType) std::pow((double) d, 1.0 / (double) MetricType::Power);
  else
    return d;
}

//! Serialize the bound object.
template<typename MetricType, typename ElemType>
template<typename Archive>
void HRectBound<MetricType, ElemType>::Serialize(Archive& ar,
                                          const unsigned int /* version */)
{
  ar & data::CreateNVP(dim, "dim");

  // Allocate memory for the bounds, if necessary.
  if (Archive::is_loading::value)
  {
    if (bounds)
      delete[] bounds;
    bounds = new math::RangeType<ElemType>[dim];
  }

  ar & data::CreateArrayNVP(bounds, dim, "bounds");
  ar & data::CreateNVP(minWidth, "minWidth");
}

} // namespace bound
} // namespace mlpack

#endif // MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
