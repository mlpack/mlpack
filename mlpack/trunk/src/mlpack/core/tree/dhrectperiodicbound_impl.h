/**
 * @file tree/dhrectperiodicbound_impl.h
 *
 * Implementation of hyper-rectangle bound policy class.
 * Template parameter t_pow is the metric to use; use 2 for Euclidian (L2).
 *
 * @experimental
 */

#ifndef TREE_DHRECTPERIODICBOUND_IMPL_H
#define TREE_DHRECTPERIODICBOUND_IMPL_H

#include <math.h>

#include "../math/math_lib.h"
#include "../io/io.h"

/**
 * Empty constructor
 */
template<int t_pow>
DHrectPeriodicBound<t_pow>::DHrectPeriodicBound() :
      box_size_(0),
      bounds_(NULL),
      dim_(0) {
}

/**
 *Specifies the box size, but not dimensionality.
 */
template<int t_pow>
DHrectPeriodicBound<t_pow>::DHrectPeriodicBound(arma::vec box) :
      bounds_(new Range[box.n_rows]),
      dim_(box.n_rows),
      box_size_(box) {
}

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set and a box with said dimensionality.
 */
template<int t_pow>
DHrectPeriodicBound<t_pow>::DHrectPeriodicBound(size_t dimension, arma::vec box) :
      box_size_(box),
      bounds_(new Range[dimension]),
      dim_(dimension) {
  mlpack::IO::AssertMessage(dim_ == ~((size_t)0), "Already initialized");
  Reset();
}

/**
 * Destructor: clean up memory
 */
template<int t_pow>
DHrectPeriodicBound<t_pow>::~DHrectPeriodicBound() {
  if(bounds_)
    delete[] bounds_;
}

/**
 * Modifies the box_size_ to the desired dimenstions.
 */
template<int t_pow>
void DHrectPeriodicBound<t_pow>::SetBoxSize(arma::vec box) {
  box_size_ = box;
}

/**
 * Returns the box_size_ vector.
 */
template<int t_pow>
arma::vec DHrectPeriodicBound<t_pow>::GetBoxSize() {
  return box_size_;
}

/**
 * Makes this (uninitialized) box the average of the two arguments,
 * i.e. the max and min of each range is the average of the maxes and mins
 * of the arguments.
 *
 * Added by: Bill March, 5/7
 */
template<int t_pow>
void DHrectPeriodicBound<t_pow>::AverageBoxesInit(const DHrectPeriodicBound& box1,
                                                  const DHrectPeriodicBound& box2) {

  dim_ = box1.dim();
  mlpack::IO::Assert(dim_ == box2.dim());

  if(bounds_)
    delete[] bounds_;
  bounds_ = new Range[dim_];

  for (size_t i = 0; i < dim_; i++) {
    Range range;
    range = box1.get(i) +  box2.get(i);
    range *= 0.5;
    bounds_[i] = range;
  }

  box_size_ = box1.GetBoxSize();
}

/**
 * Resets all dimensions to the empty set.
 */
template<int t_pow>
void DHrectPeriodicBound<t_pow>::Reset() {
  for (size_t i = 0; i < dim_; i++) {
    bounds_[i] = Range();
  }
}

/**
 * Sets the dimensionality.
 */
template<int t_pow>
void DHrectPeriodicBound<t_pow>::SetSize(size_t dim) {
  if(bounds_)
    delete[] bounds_;

  bounds_ = new Range[dim];
  dim_ = dim;
  Reset();
}

/**
 * Determines if a point is within this bound.
 */
template<int t_pow>
bool DHrectPeriodicBound<t_pow>::Contains(const arma::vec& point) const {
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
const Range DHrectPeriodicBound<t_pow>::operator[](size_t i) const {
  return bounds_[i];
}

/**
 * Sets the range for the given dimension.
 */
template<int t_pow>
Range& DHrectPeriodicBound<t_pow>::operator[](size_t i) {
  return bounds_[i];
}

/**
 * Calculates the maximum distance within the rectangle
 */
template<int t_pow>
double DHrectPeriodicBound<t_pow>::CalculateMaxDistanceSq() const {
  double max_distance = 0;
  for (size_t i = 0; i < dim_; i++)
    max_distance += pow(bounds_[i].width(), 2);

  return max_distance;
}

/** Calculates the midpoint of the range */
template<int t_pow>
void DHrectPeriodicBound<t_pow>::CalculateMidpoint(arma::vec& centroid) const {
  // set size correctly if necessary
  if(!(centroid.n_elem == dim_))
    centroid.set_size(dim_);

  for(size_t i = 0; i < dim_; i++) {
    centroid(i) = bounds_[i].mid();
  }
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<int t_pow>
double DHrectPeriodicBound<t_pow>::MinDistanceSq(const arma::vec& point) const {
  double sum = 0;

  for (size_t d = 0; d < dim_; d++){
    double a = point[d];
    double v = 0, bh;
    bh = bounds_[d].hi - bounds_[d].lo;
    bh = bh - floor(bh / box_size_[d]) * box_size_[d];
    a = a - bounds_[d].lo;
    a = a - floor(a / box_size_[d]) * box_size_[d];
    if (bh > a)
      v = std::min( a - bh, box_size_[d]-a);
    sum += pow(v, (double) t_pow);
  }

  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<int t_pow>
double DHrectPeriodicBound<t_pow>::MinDistanceSq(const DHrectPeriodicBound& other) const {
  double sum = 0;

  mlpack::IO::Assert(dim_ == other.dim_);

  for (size_t d = 0; d < dim_; d++){
    double v = 0, d1, d2, d3;
    d1 = ((bounds_[d].hi > bounds_[d].lo) | (other.bounds_[d].hi > other.bounds_[d].lo)) *
      std::min(other.bounds_[d].lo - bounds_[d].hi, bounds_[d].lo - other.bounds_[d].hi);
    d2 = ((bounds_[d].hi > bounds_[d].lo) & (other.bounds_[d].hi > other.bounds_[d].lo)) *
      std::min(other.bounds_[d].lo - bounds_[d].hi, bounds_[d].lo - other.bounds_[d].hi + box_size_[d]);
    d3 = ((bounds_[d].hi > bounds_[d].lo) & (other.bounds_[d].hi > other.bounds_[d].lo)) *
      std::min(other.bounds_[d].lo - bounds_[d].hi + box_size_[d], bounds_[d].lo - other.bounds_[d].hi);
    v = (d1 + fabs(d1)) + (d2 + fabs(d2)) + (d3 + fabs(d3));
    sum += pow(v, (double) t_pow);
  }
  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<int t_pow>
double DHrectPeriodicBound<t_pow>::MaxDistanceSq(const arma::vec& point) const {
  double sum = 0;

  for (size_t d = 0; d < dim_; d++) {
    double b = point[d];
    double v = box_size_[d] / 2.0;
    double ah, al;
    ah = bounds_[d].hi - b;
    ah = ah - floor(ah / box_size_[d]) * box_size_[d];
    if (ah < v) {
      v = ah;
    } else {
      al = bounds_[d].lo - b;
      al = al - floor(al / box_size_[d]) * box_size_[d];
      if (al > v) {
        v = (2 * v) - al;
      }
    }
    sum += pow(fabs(v), (double) t_pow);
  }
  return pow(sum, 2.0 / (double) t_pow);
}

/**
 * Computes maximum distance.
 */
template<int t_pow>
double DHrectPeriodicBound<t_pow>::MaxDistanceSq(const DHrectPeriodicBound& other) const {
  double sum = 0;

  mlpack::IO::Assert(dim_ == other.dim_);

  for (size_t d = 0; d < dim_; d++){
    double v = box_size_[d] / 2.0;
    double dh, dl;
    dh = bounds_[d].hi - other.bounds_[d].lo;
    dh = dh - floor(dh / box_size_[d]) * box_size_[d];
    dl = other.bounds_[d].hi - bounds_[d].lo;
    dl = dl - floor(dl / box_size_[d]) * box_size_[d];
    v = fabs(std::max(std::min(dh, v), std::min(dl, v)));

    sum += pow(v, (double) t_pow);
  }
  return pow(sum, 2.0 / (double) t_pow);
}



template<int t_pow>
double DHrectPeriodicBound<t_pow>::MaxDelta(const DHrectPeriodicBound& other,
                                            double box_width, int dim) const {
  double result = 0.5 * box_width;
  double temp = other.bounds_[dim].hi - bounds_[dim].lo;
  temp = temp - floor(temp / box_width) * box_width;
  if (temp > box_width / 2) {
    temp = other.bounds_[dim].lo - bounds_[dim].hi;
    temp = temp - floor(temp / box_width) * box_width;
    if (temp > box_width / 2) {
      result = other.bounds_[dim].hi - bounds_[dim].lo;
      result = result - floor(temp / box_width + 1) * box_width;
    }
  } else {
    result = temp;
  }

  return result;
}

template<int t_pow>
double DHrectPeriodicBound<t_pow>::MinDelta(const DHrectPeriodicBound& other,
                                            double box_width, int dim) const {
  double result = -0.5 * box_width;
  double temp = other.bounds_[dim].hi - bounds_[dim].lo;
  temp = temp - floor(temp / box_width) * box_width;
  if (temp > box_width / 2) {
    temp = other.bounds_[dim].hi - bounds_[dim].hi;
    temp -= floor(temp / box_width) * box_width;
    if (temp > box_width / 2)
      result = temp - box_width;

  } else {
    temp = other.bounds_[dim].hi - bounds_[dim].hi;
    result = temp - floor(temp / box_width) * box_width;
  }

  return result;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<int t_pow>
Range DHrectPeriodicBound<t_pow>::RangeDistanceSq(const DHrectPeriodicBound& other) const {
  double sum_lo = 0;
  double sum_hi = 0;

  mlpack::IO::Assert(dim_ == other.dim_);

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
Range DHrectPeriodicBound<t_pow>::RangeDistanceSq(const arma::vec& point) const {
  double sum_lo = 0;
  double sum_hi = 0;

  mlpack::IO::Assert(point.n_elem == dim_);

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
 * Calculates closest-to-their-midpoint bounding box distance,
 * i.e. calculates their midpoint and finds the minimum box-to-point
 * distance.
 *
 * Equivalent to:
 * <code>
 * other.CalcMidpoint(&other_midpoint)
 * return MinDistanceSqToPoint(other_midpoint)
 * </code>
 */
template<int t_pow>
double DHrectPeriodicBound<t_pow>::MinToMidSq(const DHrectPeriodicBound& other) const {
  double sum = 0;

  mlpack::IO::Assert(dim_ == other.dim_);

  for (size_t d = 0; d < dim_; d++) {
    double v = other.bounds_[d].mid();
    double v1 = bounds_[d].lo - v;
    double v2 = v - bounds_[d].hi;

    v = (v1 + fabs(v1)) + (v2 + fabs(v2));

    sum += pow(v, (double) t_pow); // v is non-negative
  }

  return pow(sum, 2.0 / (double) t_pow) / 4.0;
}

/**
 * Computes minimax distance, where the other node is trying to avoid me.
 */
template<int t_pow>
double DHrectPeriodicBound<t_pow>::MinimaxDistanceSq(const DHrectPeriodicBound& other) const {
  double sum = 0;

  mlpack::IO::Assert(dim_ == other.dim_);

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
double DHrectPeriodicBound<t_pow>::MidDistanceSq(const DHrectPeriodicBound& other) const {
  double sum = 0;

  mlpack::IO::Assert(dim_ == other.dim_);


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
DHrectPeriodicBound<t_pow>& DHrectPeriodicBound<t_pow>::operator|=(const arma::vec& vector) {
  mlpack::IO::Assert(vector.n_elem == dim_);

  for (size_t i = 0; i < dim_; i++) {
    bounds_[i] |= vector[i];
  }

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<int t_pow>
DHrectPeriodicBound<t_pow>& DHrectPeriodicBound<t_pow>::operator|=(const DHrectPeriodicBound& other) {
  mlpack::IO::Assert(other.dim_ == dim_);

  for (size_t i = 0; i < dim_; i++) {
    bounds_[i] |= other.bounds_[i];
  }

  return *this;
}


/**
 * Expand this bounding box to encompass another point. Done to
 * minimize added volume in periodic coordinates.
 */
template<int t_pow>
DHrectPeriodicBound<t_pow>& DHrectPeriodicBound<t_pow>::Add(const arma::vec& other,
                                                            const arma::vec& size) {
  mlpack::IO::Assert(other.n_elem == dim_);
  // Catch case of uninitialized bounds
  if (bounds_[0].hi < 0){
    for (size_t i = 0; i < dim_; i++){
      bounds_[i] |= other[i];
    }
  }

  for (size_t i= 0; i < dim_; i++){
    double ah, al;
    ah = bounds_[i].hi - other[i];
    al = bounds_[i].lo - other[i];
    ah = ah - floor(ah / size[i]) * size[i];
    al = al - floor(al / size[i]) * size[i];
    if (ah < al) {
      if (size[i] - ah < al) {
        bounds_[i].hi = other[i];
      } else {
        bounds_[i].lo = other[i];
      }
    }
  }
  return *this;
}

/**
 * Expand this bounding box in periodic coordinates, minimizing added volume.
 */
template<int t_pow>
DHrectPeriodicBound<t_pow>& DHrectPeriodicBound<t_pow>::Add(const DHrectPeriodicBound& other,
                                                            const arma::vec& size){
  if (bounds_[0].hi < 0){
    for (size_t i = 0; i < dim_; i++){
      bounds_[i] |= other.bounds_[i];
    }
  }

  for (size_t i = 0; i < dim_; i++) {
    double ah, al, bh, bl;
    ah = bounds_[i].hi;
    al = bounds_[i].lo;
    bh = other.bounds_[i].hi;
    bl = other.bounds_[i].lo;
    ah = ah - al;
    bh = bh - al;
    bl = bl - al;
    ah = ah - floor(ah / size[i]) * size[i];
    bh = bh - floor(bh / size[i]) * size[i];
    bl = bl - floor(bl / size[i]) * size[i];

    if (((bh > ah) & ((bh < bl) | (ah > bl))) ||
        ((bh >= bl) & (bl > ah) & (bh < ah -bl + size[i]))){
      bounds_[i].hi = other.bounds_[i].hi;
    }

    if (bl > ah && ((bl > bh) || (bh >= ah -bl + size[i]))){
      bounds_[i].lo = other.bounds_[i].lo;
    }

    if ((ah > bl) & (bl > bh)){
      bounds_[i].lo = 0;
      bounds_[i].hi = size[i];
    }
  }

  return *this;
}

#endif
