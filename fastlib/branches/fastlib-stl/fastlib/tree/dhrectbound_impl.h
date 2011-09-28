/**
 * @file tree/dhrectbound_impl.h
 *
 * Implementation of hyper-rectangle bound policy class.
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 *
 * @experimental
 */
#ifndef TREE_DHRECTBOUND_IMPL_H
#define TREE_DHRECTBOUND_IMPL_H

#include <math.h>

#include "../math/math_lib.h"

// In case it has not been included yet.
#include "dhrectbound.h"

namespace mlpack {
namespace bound {

/**
 * Empty constructor
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound() {
  bounds_ = NULL;
  dim_ = 0;
}

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<int t_pow>
HRectBound<t_pow>::HRectBound(size_t dimension) {
  bounds_ = new Range[dimension];

  dim_ = dimension;
  Clear();
}

/**
 * Destructor: clean up memory
 */
template<int t_pow>
HRectBound<t_pow>::~HRectBound() {
  if(bounds_)
    delete[] bounds_;
}

/**
 * Resets all dimensions to the empty set.
 */
template<int t_pow>
void HRectBound<t_pow>::Clear() {
  for (size_t i = 0; i < dim_; i++) {
    bounds_[i] = Range();
  }
}

/**
 * Sets the dimensionality.
 */
template<int t_pow>
void HRectBound<t_pow>::SetSize(size_t dim) {
  if(bounds_)
    delete[] bounds_;

  bounds_ = new Range[dim];
  dim_ = dim;
  Clear();
}

/**
 * Determines if a point is within this bound.
 */
template<int t_pow>
bool HRectBound<t_pow>::Contains(const arma::vec& point) const {
  for (size_t i = 0; i < point.n_elem; i++) {
    if (!bounds_[i].Contains(point(i))) {
      return false;
    }
  }

  return true;
}

/**
 * Gets the range for a particular dimension.
 */
template<int t_pow>
const Range HRectBound<t_pow>::operator[](size_t i) const {
  return bounds_[i];
}

/**
 * Sets the range for the given dimension.
 */
template<int t_pow>
Range& HRectBound<t_pow>::operator[](size_t i) {
  return bounds_[i];
}

/***
 * Calculates the centroid of the range, placing it into the given vector.
 *
 * @param centroid Vector which the centroid will be written to.
 */
template<int t_pow>
void HRectBound<t_pow>::Centroid(arma::vec& centroid) const {
  // set size correctly if necessary
  if(!(centroid.n_elem == dim_))
    centroid.set_size(dim_);

  for(size_t i = 0; i < dim_; i++) {
    centroid(i) = bounds_[i].mid();
  }
}

/**
 * Calculates minimum bound-to-bound squared distance, with
 * an offset between their respective coordinate systems.
 */
template<int t_pow>
double HRectBound<t_pow>::MinDistance(const HRectBound& other,
                                      const arma::vec& offset) const {
  double sum = 0;

  assert(dim_ == other.dim_);
  assert(dim_ == offset.n_elem);

  for(size_t d = 0; d < dim_; d++) {
    double v1 = other.bounds_[d].lo - offset[d] - bounds_[d].hi;
    double v2 = bounds_[d].lo + offset[d] - other.bounds_[d].lo;

    double v = (v1 + fabs(v1)) + (v2 + fabs(v2));

    sum += pow(v, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MinDistance(const arma::vec& point) const {
  assert(point.n_elem == dim_);

  double sum = 0;
  const Range* mbound = bounds_;

  double lower, higher;
  for(size_t d = 0; d < dim_; d++) {
    lower = mbound->lo - point[d]; // negative if point[d] > bounds_[d]
    higher = point[d] - mbound->hi; // negative if point[d] < bounds_[d]

    // since only one of 'lower' or 'higher' is negative, if we add each's
    // absolute value to itself and then sum those two, our result is the
    // nonnegative half of the equation times two; then we raise to power t_pow
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);

    // move bound pointer
    mbound++;
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
double HRectBound<t_pow>::MinDistance(const HRectBound& other) const {
  assert(dim_ == other.dim_);

  double sum = 0;
  const Range* mbound = bounds_;
  const Range* obound = other.bounds_;

  double lower, higher;
  for (size_t d = 0; d < dim_; d++) {
    lower = obound->lo - mbound->hi;
    higher = mbound->lo - obound->hi;
    // We invoke the following:
    //   x + fabs(x) = max(x * 2, 0)
    //   (x * 2)^2 / 4 = x^2
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);

    // move bound pointers
    mbound++;
    obound++;
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MaxDistance(const arma::vec& point) const {
  double sum = 0;

  assert(point.n_elem == dim_);

  for (size_t d = 0; d < dim_; d++) {
    double v = fabs(std::max(
      point[d] - bounds_[d].lo,
      bounds_[d].hi - point[d]));
    sum += pow(v, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Computes maximum distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MaxDistance(const HRectBound& other) const {
  double sum = 0;

  assert(dim_ == other.dim_);

  double v;
  for(size_t d = 0; d < dim_; d++) {
    v = fabs(std::max(
      other.bounds_[d].hi - bounds_[d].lo,
      bounds_[d].hi - other.bounds_[d].lo));
    sum += pow(v, (double) t_pow); // v is non-negative
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<int t_pow>
Range HRectBound<t_pow>::RangeDistance(const HRectBound& other) const {
  double sum_lo = 0;
  double sum_hi = 0;

  assert(dim_ == other.dim_);

  double v1, v2, v_lo, v_hi;
  for (size_t d = 0; d < dim_; d++) {
    v1 = other.bounds_[d].lo - bounds_[d].hi;
    v2 = bounds_[d].lo - other.bounds_[d].hi;
    // one of v1 or v2 is negative
    if(v1 >= v2) {
      v_hi = -v2; // make it nonnegative
      v_lo = (v1 > 0) ? v1 : 0; // force to be 0 if negative
    } else {
      v_hi = -v1; // make it nonnegative
      v_lo = (v2 > 0) ? v2 : 0; // force to be 0 if negative
    }

    sum_lo += pow(v_lo, (double) t_pow);
    sum_hi += pow(v_hi, (double) t_pow);
  }

  return Range(pow(sum_lo, 2.0 / (double) t_pow),
                pow(sum_hi, 2.0 / (double) t_pow));
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<int t_pow>
Range HRectBound<t_pow>::RangeDistance(const arma::vec& point) const {
  double sum_lo = 0;
  double sum_hi = 0;

  assert(point.n_elem == dim_);

  double v1, v2, v_lo, v_hi;
  for(size_t d = 0; d < dim_; d++) {
    v1 = bounds_[d].lo - point[d];
    v2 = point[d] - bounds_[d].hi;
    // one of v1 or v2 is negative
    if(v1 >= 0) {
      v_hi = -v2;
      v_lo = v1;
    } else {
      v_hi = -v1;
      v_lo = v2;
    }

    sum_lo += pow(v_lo, (double) t_pow);
    sum_hi += pow(v_hi, (double) t_pow);
  }

  return Range(pow(sum_lo, 2.0 / (double) t_pow),
                pow(sum_hi, 2.0 / (double) t_pow));
}

/**
 * Computes minimax distance, where the other node is trying to avoid me.
 */
template<int t_pow>
double HRectBound<t_pow>::MinimaxDistance(const HRectBound& other) const {
  double sum = 0;

  assert(dim_ == other.dim_);

  for(size_t d = 0; d < dim_; d++) {
    double v1 = other.bounds_[d].hi - bounds_[d].hi;
    double v2 = bounds_[d].lo - other.bounds_[d].lo;
    double v = std::max(v1, v2);
    v = (v + fabs(v)); /* truncate negatives to zero */
    sum += pow(v, (double) t_pow); // v is non-negative
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates midpoint-to-midpoint bounding box distance.
 */
template<int t_pow>
double HRectBound<t_pow>::MidDistance(const HRectBound& other) const {
  double sum = 0;

  assert(dim_ == other.dim_);

  for (size_t d = 0; d < dim_; d++) {
    // take the midpoint of each dimension (left multiplied by two for
    // calculation speed) and subtract from each other, then raise to t_pow
    sum += pow(fabs(bounds_[d].hi + bounds_[d].lo - other.bounds_[d].hi -
          other.bounds_[d].lo), (double) t_pow);
  }

  // take t_pow/2'th root and divide by constant of 4 (leftover from previous
  // step)
  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Expands this region to include a new point.
 */
template<int t_pow>
HRectBound<t_pow>& HRectBound<t_pow>::operator|=(const arma::vec& vector) {
  assert(vector.n_elem == dim_);

  for (size_t i = 0; i < dim_; i++) {
    bounds_[i] |= vector[i];
  }

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<int t_pow>
HRectBound<t_pow>& HRectBound<t_pow>::operator|=(const HRectBound& other) {
  assert(other.dim_ == dim_);

  for (size_t i = 0; i < dim_; i++) {
    bounds_[i] |= other.bounds_[i];
  }

  return *this;
}

}; // namespace bound
}; // namespace mlpack

#endif
