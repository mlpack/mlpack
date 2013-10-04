/**
 * @file periodichrectbound_impl.hpp
 *
 * Implementation of periodic hyper-rectangle bound policy class.
 * Template parameter t_pow is the metric to use; use 2 for Euclidian (L2).
 *
 * This file is part of MLPACK 1.0.7.
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
      bounds(NULL),
      dim(0),
      box(/* empty */)
{ /* nothing to do */ }

/**
 * Specifies the box size, but not dimensionality.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>::PeriodicHRectBound(arma::vec box) :
      bounds(new math::Range[box.n_rows]),
      dim(box.n_rows),
      box(box)
{ /* nothing to do */ }

/***
 * Copy constructor.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>::PeriodicHRectBound(const PeriodicHRectBound& other) :
      dim(other.Dim()),
      box(other.Box())
{
  bounds = new math::Range[other.Dim()];
  for (size_t i = 0; i < dim; i++)
    bounds[i] |= other[i];
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
  if (bounds)
    delete[] bounds;
}

/**
 * Modifies the box to the desired dimenstions.
 */
template<int t_pow>
void PeriodicHRectBound<t_pow>::SetBoxSize(arma::vec box)
{
  box = box;
}

/**
 * Resets all dimensions to the empty set.
 */
template<int t_pow>
void PeriodicHRectBound<t_pow>::Clear()
{
  for (size_t i = 0; i < dim; i++)
    bounds[i] = math::Range();
}

/**
 * Gets the range for a particular dimension.
 */
template<int t_pow>
const math::Range PeriodicHRectBound<t_pow>::operator[](size_t i) const
{
  return bounds[i];
}

/**
 * Sets the range for the given dimension.
 */
template<int t_pow>
math::Range& PeriodicHRectBound<t_pow>::operator[](size_t i)
{
  return bounds[i];
}

/** Calculates the midpoint of the range */
template<int t_pow>
void PeriodicHRectBound<t_pow>::Centroid(arma::vec& centroid) const
{
  // set size correctly if necessary
  if (!(centroid.n_elem == dim))
    centroid.set_size(dim);

  for (size_t i = 0; i < dim; i++)
    centroid(i) = bounds[i].Mid();
}

/**
 * Calculates minimum bound-to-point squared distance.
 *
 */

template<int t_pow>
double PeriodicHRectBound<t_pow>::MinDistance(const arma::vec& point) const
{
  arma::vec point2 = point;
  double totalMin = 0;
  // Create the mirrored images. The minimum distance from the bound to a
  // mirrored point is the minimum periodic distance.
  arma::vec box = box;
  for (int i = 0; i < dim; i++)
  {
    point2 = point;
    double min = 100000000;
    // Mod the point within the box.

    if (box[i] < 0)
    {
      box[i] = abs(box[i]);
    }
    if (box[i] != 0)
    {
      if (abs(point[i]) > box[i])
      {
        point2[i] = fmod(point2[i],box[i]);
      }
    }

    for (int k = 0; k < 3; k++)
    {
      arma::vec point3 = point2;

      if (k == 1)
        point3[i] += box[i];
      else if (k == 2)
        point3[i] -= box[i];

      double tempMin;
      double sum = 0;

      double lower, higher;
      lower = bounds[i].Lo() - point3[i];
      higher = point3[i] - bounds[i].Hi();

      sum += pow((lower + fabs(lower)) +
          (higher + fabs(higher)), (double) t_pow);
      tempMin = pow(sum, 2.0 / (double) t_pow) / 4.0;

      if (tempMin < min)
        min = tempMin;
    }

    totalMin += min;
  }
  return totalMin;

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
  double totalMin = 0;
  // Create the mirrored images. The minimum distance from the bound to a
  // mirrored point is the minimum periodic distance.
  arma::vec box = box;
  PeriodicHRectBound<2> a(other);

  for (int i = 0; i < dim; i++)
  {
    double min = DBL_MAX;
    if (box[i] < 0)
      box[i] = abs(box[i]);

    if (box[i] != 0)
    {
      if (abs(other[i].Lo()) > box[i])
        a[i].Lo() = fmod(a[i].Lo(),box[i]);

      if (abs(other[i].Hi()) > box[i])
        a[i].Hi() = fmod(a[i].Hi(),box[i]);
    }

    for (int k = 0; k < 3; k++)
    {
      PeriodicHRectBound<2> b = a;
      if (k == 1)
      {
        b[i].Lo() += box[i];
        b[i].Hi() += box[i];
      }
      else if (k == 2)
      {
        b[i].Lo() -= box[i];
        b[i].Hi() -= box[i];
      }

      double sum = 0;
      double tempMin;
      double sumLower = 0;
      double sumHigher = 0;

      double lower, higher, lowerLower, lowerHigher, higherLower,
              higherHigher;

      // If the bound crosses over the box, split ito two seperate bounds and
      // find the minimum distance between them.
      if (b[i].Hi() < b[i].Lo())
      {
        PeriodicHRectBound<2> d(b);
        PeriodicHRectBound<2> c(b);
        d[i].Lo() = 0;
        c[i].Hi() = box[i];

        if (k == 1)
        {
          d[i].Lo() += box[i];
          c[i].Hi() += box[i];
        }
        else if (k == 2)
        {
          d[i].Lo() -= box[i];
          c[i].Hi() -= box[i];
        }

        d[i].Hi() = b[i].Hi();
        c[i].Lo() = b[i].Lo();

        lowerLower = d[i].Lo() - bounds[i].Hi();
        higherLower = bounds[i].Lo() - d[i].Hi();

        lowerHigher = c[i].Lo() - bounds[i].Hi();
        higherHigher = bounds[i].Lo() - c[i].Hi();

        sumLower += pow((lowerLower + fabs(lowerLower)) +
                         (higherLower + fabs(higherLower)), (double) t_pow);

        sumHigher += pow((lowerHigher + fabs(lowerHigher)) +
                          (higherHigher + fabs(higherHigher)), (double) t_pow);

        if (sumLower > sumHigher)
          tempMin = pow(sumHigher, 2.0 / (double) t_pow) / 4.0;
        else
          tempMin = pow(sumLower, 2.0 / (double) t_pow) / 4.0;
      }
      else
      {
        lower = b[i].Lo() - bounds[i].Hi();
        higher = bounds[i].Lo() - b[i].Hi();
        // We invoke the following:
        //   x + fabs(x) = max(x * 2, 0)
        //   (x * 2)^2 / 4 = x^2
        sum += pow((lower + fabs(lower)) +
            (higher + fabs(higher)), (double) t_pow);
        tempMin = pow(sum, 2.0 / (double) t_pow) / 4.0;
      }

      if (tempMin < min)
        min = tempMin;
    }
    totalMin += min;
  }
  return totalMin;
}


/**
 * Calculates maximum bound-to-point squared distance.
 */
template<int t_pow>
double PeriodicHRectBound<t_pow>::MaxDistance(const arma::vec& point) const
{
  arma::vec point2 = point;
  double totalMax = 0;
  //Create the mirrored images. The minimum distance from the bound to a
  //mirrored point is the minimum periodic distance.
  arma::vec box = box;
  for (int i = 0; i < dim; i++)
  {
    point2 = point;
    double max = 0;
    // Mod the point within the box.

    if (box[i] < 0)
      box[i] = abs(box[i]);

    if (box[i] != 0)
      if (abs(point[i]) > box[i])
        point2[i] = fmod(point2[i],box[i]);

    for (int k = 0; k < 3; k++)
    {
      arma::vec point3 = point2;

      if (k == 1)
        point3[i] += box[i];
      else if (k == 2)
        point3[i] -= box[i];

      double tempMax;
      double sum = 0;

      double v = std::max(fabs(point3[i] - bounds[i].Lo()),
          fabs(bounds[i].Hi() - point3[i]));
      sum += pow(v, (double) t_pow);

      tempMax = pow(sum, 2.0 / (double) t_pow) / 4.0;

      if (tempMax > max)
        max = tempMax;
    }

    totalMax += max;
  }
  return totalMax;

}

/**
 * Computes maximum distance.
 */
template<int t_pow>
double PeriodicHRectBound<t_pow>::MaxDistance(
    const PeriodicHRectBound& other) const
{
  double totalMax = 0;
  //Create the mirrored images. The minimum distance from the bound to a
  //mirrored point is the minimum periodic distance.
  arma::vec box = box;
  PeriodicHRectBound<2> a(other);


  for (int i = 0; i < dim; i++)
  {
    double max = 0;
    if (box[i] < 0)
      box[i] = abs(box[i]);

    if (box[i] != 0)
    {
      if (abs(other[i].Lo()) > box[i])
        a[i].Lo() = fmod(a[i].Lo(),box[i]);

      if (abs(other[i].Hi()) > box[i])
        a[i].Hi() = fmod(a[i].Hi(),box[i]);
    }

    for (int k = 0; k < 3; k++)
    {
      PeriodicHRectBound<2> b = a;
      if (k == 1)
      {
        b[i].Lo() += box[i];
        b[i].Hi() += box[i];
      }
      else if (k == 2)
      {
        b[i].Lo() -= box[i];
        b[i].Hi() -= box[i];
      }

      double sum = 0;
      double tempMax;

      double sumLower = 0, sumHigher = 0;


      // If the bound corsses over the box, split ito two seperate bounds and
      // find thhe minimum distance between them.
      if (b[i].Hi() < b[i].Lo())
      {
        PeriodicHRectBound<2> d(b);
        PeriodicHRectBound<2> c(b);
        a[i].Lo() = 0;
        c[i].Hi() = box[i];

        if (k == 1)
        {
          d[i].Lo() += box[i];
          c[i].Hi() += box[i];
        }
        else if (k == 2)
        {
          d[i].Lo() -= box[i];
          c[i].Hi() -= box[i];
        }

        d[i].Hi() = b[i].Hi();
        c[i].Lo() = b[i].Lo();

        double vLower = std::max(fabs(d.bounds[i].Hi() - bounds[i].Lo()),
            fabs(bounds[i].Hi() - d.bounds[i].Lo()));

        double vHigher = std::max(fabs(c.bounds[i].Hi() - bounds[i].Lo()),
            fabs(bounds[i].Hi() - c.bounds[i].Lo()));

        sumLower += pow(vLower, (double) t_pow);
        sumHigher += pow(vHigher, (double) t_pow);

        if (sumLower > sumHigher)
          tempMax = pow(sumHigher, 2.0 / (double) t_pow) / 4.0;
        else
          tempMax = pow(sumLower, 2.0 / (double) t_pow) / 4.0;
      }
      else
      {
        double v = std::max(fabs(b.bounds[i].Hi() - bounds[i].Lo()),
            fabs(bounds[i].Hi() - b.bounds[i].Lo()));
        sum += pow(v, (double) t_pow); // v is non-negative.
        tempMax = pow(sum, 2.0 / (double) t_pow);
      }


      if (tempMax > max)
        max = tempMax;
    }
    totalMax += max;
  }
  return totalMax;
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

  Log::Assert(point.n_elem == dim);

  double v1, v2, v_lo, v_hi;
  for (size_t d = 0; d < dim; d++)
  {
    v1 = bounds[d].Lo() - point[d];
    v2 = point[d] - bounds[d].Hi();
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

  Log::Assert(dim == other.dim);

  double v1, v2, v_lo, v_hi;
  for (size_t d = 0; d < dim; d++)
  {
    v1 = other.bounds[d].Lo() - bounds[d].Hi();
    v2 = bounds[d].Lo() - other.bounds[d].Hi();
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
 * Expands this region to include a new point.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>& PeriodicHRectBound<t_pow>::operator|=(
    const arma::vec& vector)
{
  Log::Assert(vector.n_elem == dim);

  for (size_t i = 0; i < dim; i++)
    bounds[i] |= vector[i];

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<int t_pow>
PeriodicHRectBound<t_pow>& PeriodicHRectBound<t_pow>::operator|=(
    const PeriodicHRectBound& other)
{
  Log::Assert(other.dim == dim);

  for (size_t i = 0; i < dim; i++)
    bounds[i] |= other.bounds[i];

  return *this;
}

/**
 * Determines if a point is within this bound.
 */
template<int t_pow>
bool PeriodicHRectBound<t_pow>::Contains(const arma::vec& point) const
{
  for (size_t i = 0; i < point.n_elem; i++)
    if (!bounds[i].Contains(point(i)))
      return false;

  return true;
}

/**
 * Returns a string representation of this object.
 */
template<int t_pow>
std::string PeriodicHRectBound<t_pow>::ToString() const
{
  std::ostringstream convert;
  convert << "PeriodicHRectBound [" << this << "]" << std::endl;
  convert << "bounds: " << bounds->ToString() << std::endl;
  convert << "dim: " << dim << std::endl;
  convert << "box: " << box;
  return convert.str();
}

}; // namespace bound
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_PERIODICHRECTBOUND_IMPL_HPP
