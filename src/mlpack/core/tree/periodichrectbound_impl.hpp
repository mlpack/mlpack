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
PeriodicHRectBound<t_pow>::PeriodicHRectBound(const PeriodicHRectBound& other) :
      dim_(other.Dim()),
      box_(other.box())
{
  bounds_ = new math::Range[other.Dim()];
  for (size_t i = 0; i < dim_; i++) {
    bounds_[i] |= other[i];
  }
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
 *
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
}*/

template<int t_pow>
double PeriodicHRectBound<t_pow>::MinDistance(const arma::vec& point) const
{
  arma::vec point2 = point;
  double total_min = 0;
  //Create the mirrored images. The minimum distance from the bound to a
  //mirrored point is the minimum periodic distance.
  arma::vec box = box_;
  for (int i = 0; i < dim_; i++){
    point2 = point;
    double min = 100000000;
    //Mod the point within the box

    if (box[i] < 0){
      box[i] = abs(box[i]);
    }
    if (box[i] != 0){
      if (abs(point[i]) > box[i]) {
        point2[i] = fmod(point2[i],box[i]);
      }
    }

    for (int k = 0; k < 3; k++){
      arma::vec point3 = point2;
      if (k == 1) {
        point3[i] += box[i];
      }
      else if (k == 2) {
        point3[i] -= box[i];
      }

      double temp_min;
      double sum = 0;

      double lower, higher;
      lower = bounds_[i].Lo() - point3[i];
      higher = point3[i] - bounds_[i].Hi();

      sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);
      temp_min = pow(sum, 2.0 / (double) t_pow) / 4.0;

      if( temp_min < min)
        min = temp_min;
    }

    total_min += min;
  }
  return total_min;

}

/**
 * Calculates minimum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistance(other) for minimum squared distance.
 *

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
}*/

template<int t_pow>
double PeriodicHRectBound<t_pow>::MinDistance(
    const PeriodicHRectBound& other) const
{
  double total_min = 0;
  //Create the mirrored images. The minimum distance from the bound to a
  //mirrored point is the minimum periodic distance.
  arma::vec box = box_;
  PeriodicHRectBound<2> a(other);


  for (int i = 0; i < dim_; i++){
    double min = 100000000;
    if (box[i] < 0) {
      box[i] = abs(box[i]);
    }
    if (box[i] != 0) {
      if (abs(other[i].Lo()) > box[i]) {
        a[i].Lo() = fmod(a[i].Lo(),box[i]);
      }
      if (abs(other[i].Hi()) > box[i]) {
        a[i].Hi() = fmod(a[i].Hi(),box[i]);
      }
    }

    for (int k = 0; k < 3; k++){
      PeriodicHRectBound<2> b = a;
      if (k == 1) {
        b[i].Lo() += box[i];
        b[i].Hi() += box[i];
      }
      else if (k == 2) {
        b[i].Lo() -= box[i];
        b[i].Hi() -= box[i];
      }

      double sum = 0;
      double temp_min;
      double sum_lower = 0;
      double sum_higher = 0;
//      math::Range mbound = bounds_[i];
//      math::Range obound = b[i];

      double lower, higher, lower_lower, lower_higher, higher_lower,
              higher_higher;

      //If the bound corsses over the box, split ito two seperate bounds and
      //find thhe minimum distance between them.
      if( b[i].Hi() < b[i].Lo()) {
        PeriodicHRectBound<2> a(b);
        PeriodicHRectBound<2> c(b);
        double upper = b[i].Hi();
        double low = b[i].Lo();
        a[i].Lo() = 0;
        c[i].Hi() = box[i];

        if (k == 1) {
          a[i].Lo() += box[i];
          c[i].Hi() += box[i];
        }
        else if (k == 2) {
          a[i].Lo() -= box[i];
          c[i].Hi() -= box[i];
        }
        a[i].Hi() = b[i].Hi();
        c[i].Lo() = b[i].Lo();

        lower_lower = a[i].Lo() - bounds_[i].Hi();
        higher_lower = bounds_[i].Lo() - a[i].Hi();


        lower_higher = c[i].Lo() - bounds_[i].Hi();
        higher_higher = bounds_[i].Lo() - c[i].Hi();

        sum_lower += pow((lower_lower + fabs(lower_lower)) +
                         (higher_lower + fabs(higher_lower)), (double) t_pow);

        sum_higher += pow((lower_higher + fabs(lower_higher)) +
                          (higher_higher + fabs(higher_higher)), (double) t_pow);

        if (sum_lower > sum_higher) {
          temp_min = pow(sum_higher, 2.0 / (double) t_pow) / 4.0;
        }
        else {
          temp_min = pow(sum_lower, 2.0 / (double) t_pow) / 4.0;
        }

      }
      else {
        lower = b[i].Lo() - bounds_[i].Hi();
        higher = bounds_[i].Lo() - b[i].Hi();
        // We invoke the following:
        //   x + fabs(x) = max(x * 2, 0)
        //   (x * 2)^2 / 4 = x^2
        sum += pow((lower + fabs(lower)) + (higher + fabs(higher)), (double) t_pow);
        temp_min = pow(sum, 2.0 / (double) t_pow) / 4.0;
      }


      if (temp_min < min)
        min = temp_min;
    }
    total_min += min;
  }
  return total_min;
}


/**
 * Calculates maximum bound-to-point squared distance.
 */
template<int t_pow>
double PeriodicHRectBound<t_pow>::MaxDistance(const arma::vec& point) const
{
  arma::vec point2 = point;
  double total_max = 0;
  //Create the mirrored images. The minimum distance from the bound to a
  //mirrored point is the minimum periodic distance.
  arma::vec box = box_;
  for (int i = 0; i < dim_; i++){
    point2 = point;
    double max = 0;
    //Mod the point within the box

    if (box[i] < 0){
      box[i] = abs(box[i]);
    }
    if (box[i] != 0){
      if (abs(point[i]) > box[i]) {
        point2[i] = fmod(point2[i],box[i]);
      }
    }

    for (int k = 0; k < 3; k++){
      arma::vec point3 = point2;
      if (k == 1) {
        point3[i] += box[i];
      }
      else if (k == 2) {
        point3[i] -= box[i];
      }

      double temp_max;
      double sum = 0;

      double lower, higher;
      lower = bounds_[i].Lo() - point3[i];
      higher = point3[i] - bounds_[i].Hi();

      double v = fabs(std::max(point[i] - bounds_[i].Lo(),
                             bounds_[i].Hi() - point[i]));
      sum += pow(v, (double) t_pow);

      //sum += pow((lower + fabs(lower)) + (higher + fabs(higher)),
      //            (double) t_pow);
      temp_max = pow(sum, 2.0 / (double) t_pow) / 4.0;

      if (temp_max > max)
        max = temp_max;
    }

    total_max += max;
  }
  return total_max;

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
