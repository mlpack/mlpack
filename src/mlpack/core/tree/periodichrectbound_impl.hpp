/**
 * @file tree/periodichrectbound_impl.h
 *
 * Implementation of periodic hyper-rectangle bound policy class.
 * Template parameter t_pow is the metric to use; use 2 for Euclidian (L2).
 */
#ifndef __MLPACK_CORE_TREE_PERIODICHRECTBOUND_IMPL_HPP
#define __MLPACK_CORE_TREE_PERIODICHRECTBOUND_IMPL_HPP

// In case it has not already been included.
#include "periodichrectbound.hpp"

#include <math.h>

namespace mlpack {
namespace bound {

/**
 * Empty constructor
 */
template<int t_pow>
PeriodicHRectBound<t_pow>::PeriodicHRectBound() :
      bounds_(NULL),
      dim_(0),
      box_(/* empty */)
{ /* nothing to do */ }

/**
 * Specifies the box size, but not dimensionality.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>::PeriodicHRectBound(arma::vec box) :
      bounds_(new math::Range[box.n_rows]),
      dim_(box.n_rows),
      box_(box)
{ /* nothing to do */ }

/***
 * Copy constructor.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>::PeriodicHRectBound(const PeriodicHRectBound& other)
{
  // not done yet
}

/***
 * Copy operator.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>& PeriodicHRectBound<t_pow>::operator=(
    const PeriodicHRectBound& other)
{
  // not done yet

  return *this;
}

/**
 * Destructor: clean up memory
 */
template<int t_pow>
PeriodicHRectBound<t_pow>::~PeriodicHRectBound()
{
  if(bounds_)
    delete[] bounds_;
}

/**
 * Modifies the box_ to the desired dimenstions.
 */
template<int t_pow>
void PeriodicHRectBound<t_pow>::SetBoxSize(arma::vec box)
{
  box_ = box;
}

/**
 * Resets all dimensions to the empty set.
 */
template<int t_pow>
void PeriodicHRectBound<t_pow>::Clear()
{
  for (size_t i = 0; i < dim_; i++)
    bounds_[i] = math::Range();
}

/**
 * Gets the range for a particular dimension.
 */
template<int t_pow>
const math::Range PeriodicHRectBound<t_pow>::operator[](size_t i) const
{
  return bounds_[i];
}

/**
 * Sets the range for the given dimension.
 */
template<int t_pow>
math::Range& PeriodicHRectBound<t_pow>::operator[](size_t i)
{
  return bounds_[i];
}

/** Calculates the midpoint of the range */
template<int t_pow>
void PeriodicHRectBound<t_pow>::Centroid(arma::vec& centroid) const
{
  // set size correctly if necessary
  if (!(centroid.n_elem == dim_))
    centroid.set_size(dim_);

  for (size_t i = 0; i < dim_; i++)
    centroid(i) = bounds_[i].Mid();
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<int t_pow>
double PeriodicHRectBound<t_pow>::MinDistance(const arma::vec& point) const
{
  double sum = 0;

  for (size_t d = 0; d < dim_; d++)
  {
    double a = point[d];
    double v = 0, bh;
    bh = bounds_[d].Hi() - bounds_[d].Lo();
    bh = bh - floor(bh / box_[d]) * box_[d];
    a = a - bounds_[d].Lo();
    a = a - floor(a / box_[d]) * box_[d];

    if (bh > a)
      v = std::min( a - bh, box_[d]-a);

    sum += pow(v, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistance(other) for minimum squared distance.
 */
template<int t_pow>
double PeriodicHRectBound<t_pow>::MinDistance(
    const PeriodicHRectBound& other) const
{
  double sum = 0;

  Log::Assert(dim_ == other.dim_);

  for (size_t d = 0; d < dim_; d++){
    double v = 0, d1, d2, d3;
    d1 = ((bounds_[d].Hi() > bounds_[d].Lo()) |
          (other.bounds_[d].Hi() > other.bounds_[d].Lo())) *
        std::min(other.bounds_[d].Lo() - bounds_[d].Hi(),
                 bounds_[d].Lo() - other.bounds_[d].Hi());
    d2 = ((bounds_[d].Hi() > bounds_[d].Lo()) &
          (other.bounds_[d].Hi() > other.bounds_[d].Lo())) *
        std::min(other.bounds_[d].Lo() - bounds_[d].Hi(),
                 bounds_[d].Lo() - other.bounds_[d].Hi() + box_[d]);
    d3 = ((bounds_[d].Hi() > bounds_[d].Lo()) &
          (other.bounds_[d].Hi() > other.bounds_[d].Lo())) *
        std::min(other.bounds_[d].Lo() - bounds_[d].Hi() + box_[d],
                 bounds_[d].Lo() - other.bounds_[d].Hi());

    v = (d1 + fabs(d1)) + (d2 + fabs(d2)) + (d3 + fabs(d3));

    sum += pow(v, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<int t_pow>
double PeriodicHRectBound<t_pow>::MaxDistance(const arma::vec& point) const
{
  double sum = 0;

  for (size_t d = 0; d < dim_; d++)
  {
    double b = point[d];
    double v = box_[d] / 2.0;
    double ah, al;

    ah = bounds_[d].Hi() - b;
    ah = ah - floor(ah / box_[d]) * box_[d];

    if (ah < v)
    {
      v = ah;
    }
    else
    {
      al = bounds_[d].Lo() - b;
      al = al - floor(al / box_[d]) * box_[d];

      if (al > v)
        v = (2 * v) - al;
    }

    sum += pow(fabs(v), (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Computes maximum distance.
 */
template<int t_pow>
double PeriodicHRectBound<t_pow>::MaxDistance(
    const PeriodicHRectBound& other) const
{
  double sum = 0;

  Log::Assert(dim_ == other.dim_);

  for (size_t d = 0; d < dim_; d++)
  {
    double v = box_[d] / 2.0;
    double dh, dl;

    dh = bounds_[d].Hi() - other.bounds_[d].Lo();
    dh = dh - floor(dh / box_[d]) * box_[d];
    dl = other.bounds_[d].Hi() - bounds_[d].Lo();
    dl = dl - floor(dl / box_[d]) * box_[d];
    v = fabs(std::max(std::min(dh, v), std::min(dl, v)));

    sum += pow(v, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<int t_pow>
math::Range PeriodicHRectBound<t_pow>::RangeDistance(
    const arma::vec& point) const
{
  double sum_lo = 0;
  double sum_hi = 0;

  Log::Assert(point.n_elem == dim_);

  double v1, v2, v_lo, v_hi;
  for (size_t d = 0; d < dim_; d++)
  {
    v1 = bounds_[d].Lo() - point[d];
    v2 = point[d] - bounds_[d].Hi();
    // One of v1 or v2 is negative.
    if (v1 >= 0)
    {
      v_hi = -v2;
      v_lo = v1;
    }
    else
    {
      v_hi = -v1;
      v_lo = v2;
    }

    sum_lo += pow(v_lo, (double) t_pow);
    sum_hi += pow(v_hi, (double) t_pow);
  }

  return math::Range(pow(sum_lo, 2.0 / (double) t_pow),
      pow(sum_hi, 2.0 / (double) t_pow));
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<int t_pow>
math::Range PeriodicHRectBound<t_pow>::RangeDistance(
    const PeriodicHRectBound& other) const
{
  double sum_lo = 0;
  double sum_hi = 0;

  Log::Assert(dim_ == other.dim_);

  double v1, v2, v_lo, v_hi;
  for (size_t d = 0; d < dim_; d++)
  {
    v1 = other.bounds_[d].Lo() - bounds_[d].Hi();
    v2 = bounds_[d].Lo() - other.bounds_[d].Hi();
    // One of v1 or v2 is negative.
    if(v1 >= v2)
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
 * Expands this region to include a new point.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>& PeriodicHRectBound<t_pow>::operator|=(
    const arma::vec& vector)
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
PeriodicHRectBound<t_pow>& PeriodicHRectBound<t_pow>::operator|=(
    const PeriodicHRectBound& other)
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
bool PeriodicHRectBound<t_pow>::Contains(const arma::vec& point) const
{
  for (size_t i = 0; i < point.n_elem; i++)
    if (!bounds_[i].Contains(point(i)))
      return false;

  return true;
}

}; // namespace bound
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_PERIODICHRECTBOUND_IMPL_HPP
