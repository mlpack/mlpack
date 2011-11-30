/**
 * @file hrectbound_impl.hpp
 *
 * Implementation of hyper-rectangle bound policy class.
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 *
 * @experimental
 */
#ifndef __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
#define __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP

#include <math.h>

#include "../math/math_misc.hpp"

// In case it has not been included yet.
#include "hrectbound.hpp"

namespace mlpack {
namespace bound {

/**
 * Empty constructor
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound() :
    dim_(0),
    bounds_(NULL)
{ /* nothing to do */ }

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound(size_t dimension) :
    dim_(dimension),
    bounds_(new math::Range[dim_])
{ /* nothing to do */ }

/***
 * Copy constructor necessary to prevent memory leaks.
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound(const HRectBound& other) :
    dim_(other.dim()),
    bounds_(new math::Range[dim_])
{
  // Copy other bounds over.
  for (size_t i = 0; i < dim_; i++)
    bounds_[i] = other[i];
}

/***
 * Same as the copy constructor.
 */
template<int t_pow>
HRectBound<t_pow>& HRectBound<t_pow>::operator=(const HRectBound& other)
{
  if (bounds_)
    delete[] bounds_;

  // We can't just copy the bounds_ pointer like the default copy constructor
  // will!
  dim_ = other.dim();
  bounds_ = new math::Range[dim_];
  for (size_t i = 0; i < dim_; i++)
    bounds_[i] = other[i];

  return *this;
}

/**
 * Destructor: clean up memory
 */
template<int t_pow>
HRectBound<t_pow>::~HRectBound()
{
  if (bounds_)
    delete[] bounds_;
}

/**
 * Resets all dimensions to the empty set.
 */
template<int t_pow>
void HRectBound<t_pow>::Clear()
{
  for (size_t i = 0; i < dim_; i++)
    bounds_[i] = math::Range();
}

/**
 * Gets the range for a particular dimension.
 */
template<int t_pow>
const math::Range HRectBound<t_pow>::operator[](size_t i) const
{
  return bounds_[i];
}

/**
 * Sets the range for the given dimension.
 */
template<int t_pow>
math::Range& HRectBound<t_pow>::operator[](size_t i)
{
  return bounds_[i];
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
  if (!(centroid.n_elem == dim_))
    centroid.set_size(dim_);

  for (size_t i = 0; i < dim_; i++)
    centroid(i) = bounds_[i].mid();
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MinDistance(const arma::vec& point) const
{
  Log::Assert(point.n_elem == dim_);

  double sum = 0;

  double lower, higher;
  for (size_t dimension = 0; dimension < dim_; dimension++)
  {
    lower = bounds_[dimension].lo - point[dimension];
    higher = point[dimension] - bounds_[dimension].hi;

    // since only one of 'lower' or 'higher' is negative, if we add each's
    // absolute value to itself and then sum those two, our result is the
    // nonnegative half of the equation times two; then we raise to power t_pow
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);
  }

  // now take the t_pow'th root (but make sure our result is squared); then
  // divide by four to cancel out the constant of 2 (which has been squared now)
  // that was introduced earlier
  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates minimum bound-to-point squared distance, filtered
 *   by indices.
 */
template<int t_pow>
double HRectBound<t_pow>::MinDistance(const arma::vec& point,
                                      const std::vector<size_t>& indices) const
{
  Log::Assert(point.n_elem == dim_);

  double sum = 0.0;

  double lower, higher;
  for (size_t index = 0; index < indices.size(); index++)
  {
    size_t dimension = indices[index];
    lower = bounds_[dimension].lo - point[dimension];
    higher = point[dimension] - bounds_[dimension].hi;

    // since at least one of 'lower' or 'higher' is negative, if we add each's
    // absolute value to itself and then sum those two, our result is the
    // nonnegative half of the equation times two; then we raise to power t_pow
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);
  }

  // now take the t_pow'th root (but make sure our result is squared); then
  // divide by four to cancel out the constant of 2 (which has been squared now)
  // that was introduced earlier
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
  Log::Assert(dim_ == other.dim_);

  double sum = 0.0;
  double lower = 0.0;
  double higher = 0.0;

  for(size_t dimension = 0; dimension < dim_; dimension++)
  {
    lower = bounds_[dimension].lo - other.bounds_[dimension].hi;
    higher = other.bounds_[dimension].lo - bounds_[dimension].hi;

    // since at least one of 'lower' or 'higher' is negative, if we add each's
    // absolute value to itself and then sum those two, our result is the
    // nonnegative half of the equation times two; then we raise to power t_pow
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}
/**
 * Calculates minimum bound-to-bound squared distance,
 *   filtered by indices.
 *
 * Example: bound1.MinDistanceSq(other, indices) for minimum squared distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MinDistance(const HRectBound& other,
                                      const std::vector<size_t>& indices) const
{
  Log::Assert(dim_ == other.dim_);

  double sum = 0.0;
  double lower = 0.0;
  double higher = 0.0;

  for(size_t index = 0; index < indices.size(); index++)
  {
    size_t dimension = indices[index];
    lower = bounds_[dimension].lo - other.bounds_[dimension].hi;
    higher = other.bounds_[dimension].lo - bounds_[dimension].hi;

    // since only one of 'lower' or 'higher' is negative, if we add each's
    // absolute value to itself and then sum those two, our result is the
    // nonnegative half of the equation times two; then we raise to power t_pow
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MaxDistance(const arma::vec& point) const
{
  double sum = 0.0;
  double lower, higher;

  Log::Assert(point.n_elem == dim_);

  for (size_t dimension = 0; dimension < dim_; dimension++)
  {
    lower = fabs(point[dimension] - bounds_[dimension].lo);
    higher = fabs(point[dimension] - bounds_[dimension].hi);

    sum += pow(fabs(higher-lower) + higher + lower, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}
/**
 * Calculates maximum bound-to-point squared distance,
 *   filtered by indices.
 */
template<int t_pow>
double HRectBound<t_pow>::MaxDistance(const arma::vec& point,
                                      const std::vector<size_t>& indices) const
{
  double sum = 0.0;
  double lower, higher;

  Log::Assert(point.n_elem == dim_);

  for (size_t index = 0; index < indices.size(); index++)
  {
    size_t dimension = indices[index];
    lower = fabs(point[dimension] - bounds_[dimension].lo);
    higher = fabs(point[dimension] - bounds_[dimension].hi);

    sum += pow(fabs(higher-lower) + higher + lower, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Computes maximum distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MaxDistance(const HRectBound& other) const
{
  double sum = 0.0;
  double lower, higher;

  Log::Assert(other.dim_ == dim_);

  for (size_t dimension = 0; dimension < dim_; dimension++)
  {
    lower = fabs(other.bounds_[dimension].hi - bounds_[dimension].lo);
    higher = fabs(other.bounds_[dimension].lo - bounds_[dimension].hi);

    sum += pow(fabs(higher-lower) + higher + lower, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Computes the maximum distance between blocks,
 *   filtered by indices.
 */
template<int t_pow>
double HRectBound<t_pow>::MaxDistance(const HRectBound& other,
                                      const std::vector<size_t>& indices) const
{
  double sum = 0.0;
  double lower, higher;

  Log::Assert(other.dim_ == dim_);

  for (size_t index = 0; index < indices.size(); index++)
  {
    size_t dimension = indices[index];
    lower = fabs(other.bounds_[dimension].hi - bounds_[dimension].lo);
    higher = fabs(other.bounds_[dimension].lo - bounds_[dimension].hi);

    sum += pow(fabs(higher-lower) + higher + lower, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}
/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<int t_pow>
math::Range HRectBound<t_pow>::RangeDistance(const HRectBound& other) const
{
  double sum_lo = 0;
  double sum_hi = 0;

  Log::Assert(dim_ == other.dim_);

  double v1, v2, v_lo, v_hi;
  for (size_t d = 0; d < dim_; d++)
  {
    v1 = other.bounds_[d].lo - bounds_[d].hi;
    v2 = bounds_[d].lo - other.bounds_[d].hi;
    // One of v1 or v2 is negative.
    if (v1 >= v2)
    {
      v_hi = -v2; // Make it nonnegative.
      v_lo = (v1 > 0) ? v1 : 0; // Force to be 0 if negative.
    }
    else
    {
      v_hi = -v1; // Make it nonnegative.
      v_lo = (v2 > 0) ? v2 : 0; // Force to be 0 if negative.
    }

    sum_lo += pow(v_lo, (double) t_pow);
    sum_hi += pow(v_hi, (double) t_pow);
  }

  return math::Range(pow(sum_lo, 2.0 / (double) t_pow),
      pow(sum_hi, 2.0 / (double) t_pow));
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<int t_pow>
math::Range HRectBound<t_pow>::RangeDistance(const arma::vec& point) const
{
  double sum_lo = 0;
  double sum_hi = 0;

  Log::Assert(point.n_elem == dim_);

  double v1, v2, v_lo, v_hi;
  for(size_t d = 0; d < dim_; d++)
  {
    v1 = bounds_[d].lo - point[d]; // Negative if point[d] > lo.
    v2 = point[d] - bounds_[d].hi; // Negative if point[d] < hi.
    // One of v1 or v2 (or both) is negative.
    if (v1 >= 0) // point[d] <= bounds_[d].lo.
    {
      v_hi = -v2; // v2 will be larger but must be negated.
      v_lo = v1;
    }
    else // point[d] is between lo and hi, or greater than hi.
    {
      if (v2 >= 0)
      {
        v_hi = -v1; // v1 will be larger, but must be negated.
        v_lo = v2;
      }
      else
      {
        v_hi = -std::min(v1, v2); // Both are negative, but we need the larger.
        v_lo = 0;
      }
    }

    sum_lo += pow(v_lo, (double) t_pow);
    sum_hi += pow(v_hi, (double) t_pow);
  }

  return math::Range(pow(sum_lo, 2.0 / (double) t_pow),
      pow(sum_hi, 2.0 / (double) t_pow));
}

/**
 * Expands this region to include a new point.
 */
template<int t_pow>
HRectBound<t_pow>& HRectBound<t_pow>::operator|=(const arma::vec& vector)
{
  Log::Assert(vector.n_elem == dim_);

  for (size_t i = 0; i < dim_; i++)
    bounds_[i] |= vector[i];

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<int t_pow>
HRectBound<t_pow>& HRectBound<t_pow>::operator|=(const HRectBound& other)
{
  Log::Assert(other.dim_ == dim_);

  for (size_t i = 0; i < dim_; i++)
    bounds_[i] |= other.bounds_[i];

  return *this;
}

/**
 * Determines if a point is within this bound.
 */
template<int t_pow>
bool HRectBound<t_pow>::Contains(const arma::vec& point) const
{
  for (size_t i = 0; i < point.n_elem; i++)
  {
    if (!bounds_[i].Contains(point(i)))
      return false;
  }

  return true;
}

}; // namespace bound
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
