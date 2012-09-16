/**
 * @file hrectbound_impl.hpp
 *
 * Implementation of hyper-rectangle bound policy class.
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 *
 * @experimental
 *
 * This file is part of MLPACK 1.0.3.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
#define __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP

#include <math.h>

// In case it has not been included yet.
#include "hrectbound.hpp"

namespace mlpack {
namespace bound {

/**
 * Empty constructor
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound() :
    dim(0),
    bounds(NULL)
{ /* nothing to do */ }

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound(const size_t dimension) :
    dim(dimension),
    bounds(new math::Range[dim])
{ /* nothing to do */ }

/***
 * Copy constructor necessary to prevent memory leaks.
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound(const HRectBound& other) :
    dim(other.Dim()),
    bounds(new math::Range[dim])
{
  // Copy other bounds over.
  for (size_t i = 0; i < dim; i++)
    bounds[i] = other[i];
}

/***
 * Same as the copy constructor.
 */
template<int t_pow>
HRectBound<t_pow>& HRectBound<t_pow>::operator=(const HRectBound& other)
{
  if (bounds)
    delete[] bounds;

  // We can't just copy the bounds_ pointer like the default copy constructor
  // will!
  dim = other.Dim();
  bounds = new math::Range[dim];
  for (size_t i = 0; i < dim; i++)
    bounds[i] = other[i];

  return *this;
}

/**
 * Destructor: clean up memory
 */
template<int t_pow>
HRectBound<t_pow>::~HRectBound()
{
  if (bounds)
    delete[] bounds;
}

/**
 * Resets all dimensions to the empty set.
 */
template<int t_pow>
void HRectBound<t_pow>::Clear()
{
  for (size_t i = 0; i < dim; i++)
    bounds[i] = math::Range();
}

/**
 * Gets the range for a particular dimension.
 */
template<int t_pow>
inline const math::Range& HRectBound<t_pow>::operator[](const size_t i) const
{
  return bounds[i];
}

/**
 * Sets the range for the given dimension.
 */
template<int t_pow>
inline math::Range& HRectBound<t_pow>::operator[](const size_t i)
{
  return bounds[i];
}

/***
 * Calculates the centroid of the range, placing it into the given vector.
 *
 * @param centroid Vector which the centroid will be written to.
 */
template<int t_pow>
void HRectBound<t_pow>::Centroid(arma::vec& centroid) const
{
  // set size correctly if necessary
  if (!(centroid.n_elem == dim))
    centroid.set_size(dim);

  for (size_t i = 0; i < dim; i++)
    centroid(i) = bounds[i].Mid();
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<int t_pow>
template<typename VecType>
double HRectBound<t_pow>::MinDistance(const VecType& point) const
{
  Log::Assert(point.n_elem == dim);

  double sum = 0;

  double lower, higher;
  for (size_t d = 0; d < dim; d++)
  {
    lower = bounds[d].Lo() - point[d];
    higher = point[d] - bounds[d].Hi();

    // Since only one of 'lower' or 'higher' is negative, if we add each's
    // absolute value to itself and then sum those two, our result is the
    // nonnegative half of the equation times two; then we raise to power t_pow.
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);
  }

  // Now take the t_pow'th root (but make sure our result is squared); then
  // divide by four to cancel out the constant of 2 (which has been squared now)
  // that was introduced earlier.
  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates minimum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MinDistance(const HRectBound& other) const
{
  Log::Assert(dim == other.dim);

  double sum = 0;
  const math::Range* mbound = bounds;
  const math::Range* obound = other.bounds;

  double lower, higher;
  for (size_t d = 0; d < dim; d++)
  {
    lower = obound->Lo() - mbound->Hi();
    higher = mbound->Lo() - obound->Hi();
    // We invoke the following:
    //   x + fabs(x) = max(x * 2, 0)
    //   (x * 2)^2 / 4 = x^2
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);

    // Move bound pointers.
    mbound++;
    obound++;
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<int t_pow>
template<typename VecType>
double HRectBound<t_pow>::MaxDistance(const VecType& point) const
{
  double sum = 0;

  Log::Assert(point.n_elem == dim);

  for (size_t d = 0; d < dim; d++)
  {
		double v = std::max(fabs(point[d] - bounds[d].Lo()),
					       fabs(bounds[d].Hi() - point[d]));
    sum += pow(v, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Computes maximum distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MaxDistance(const HRectBound& other) const
{
  double sum = 0;

  Log::Assert(dim == other.dim);

  double v;
  for (size_t d = 0; d < dim; d++)
  {
		v = std::max(fabs(other.bounds[d].Hi() - bounds[d].Lo()),
					       fabs(bounds[d].Hi() - other.bounds[d].Lo()));
    sum += pow(v, (double) t_pow); // v is non-negative.
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<int t_pow>
math::Range HRectBound<t_pow>::RangeDistance(const HRectBound& other) const
{
  double loSum = 0;
  double hiSum = 0;

  Log::Assert(dim == other.dim);

  double v1, v2, vLo, vHi;
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

    loSum += pow(vLo, (double) t_pow);
    hiSum += pow(vHi, (double) t_pow);
  }

  return math::Range(pow(loSum, 2.0 / (double) t_pow),
                     pow(hiSum, 2.0 / (double) t_pow));
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<int t_pow>
template<typename VecType>
math::Range HRectBound<t_pow>::RangeDistance(const VecType& point) const
{
  double loSum = 0;
  double hiSum = 0;

  Log::Assert(point.n_elem == dim);

  double v1, v2, vLo, vHi;
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

    loSum += pow(vLo, (double) t_pow);
    hiSum += pow(vHi, (double) t_pow);
  }

  return math::Range(pow(loSum, 2.0 / (double) t_pow),
                     pow(hiSum, 2.0 / (double) t_pow));
}

/**
 * Expands this region to include a new point.
 */
template<int t_pow>
template<typename MatType>
HRectBound<t_pow>& HRectBound<t_pow>::operator|=(const MatType& data)
{
  Log::Assert(data.n_rows == dim);

  arma::vec mins = min(data, 1);
  arma::vec maxs = max(data, 1);

  for (size_t i = 0; i < dim; i++)
    bounds[i] |= math::Range(mins[i], maxs[i]);

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<int t_pow>
HRectBound<t_pow>& HRectBound<t_pow>::operator|=(const HRectBound& other)
{
  assert(other.dim == dim);

  for (size_t i = 0; i < dim; i++)
    bounds[i] |= other.bounds[i];

  return *this;
}

/**
 * Determines if a point is within this bound.
 */
template<int t_pow>
template<typename VecType>
bool HRectBound<t_pow>::Contains(const VecType& point) const
{
  for (size_t i = 0; i < point.n_elem; i++)
  {
    if (!bounds[i].Contains(point(i)))
      return false;
  }

  return true;
}

}; // namespace bound
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
