/**
 * @file hrectbound_impl.hpp
 *
 * Implementation of hyper-rectangle bound policy class.
 * Template parameter Power is the metric to use; use 2 for Euclidean (L2).
 *
 * @experimental
 */
#ifndef __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
#define __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP

#include <math.h>

// In case it has not been included yet.
#include "hrectbound.hpp"

namespace mlpack {
namespace bound {

/**
 * Empty constructor.
 */
template<typename MetricType>
inline HRectBound<MetricType>::HRectBound() :
    dim(0),
    bounds(NULL),
    minWidth(0)
{ /* Nothing to do. */ }

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<typename MetricType>
inline HRectBound<MetricType>::HRectBound(const size_t dimension) :
    dim(dimension),
    bounds(new math::Range[dim]),
    minWidth(0)
{ /* Nothing to do. */ }

/**
 * Copy constructor necessary to prevent memory leaks.
 */
template<typename MetricType>
inline HRectBound<MetricType>::HRectBound(const HRectBound& other) :
    dim(other.Dim()),
    bounds(new math::Range[dim]),
    minWidth(other.MinWidth())
{
  // Copy other bounds over.
  for (size_t i = 0; i < dim; i++)
    bounds[i] = other[i];
}

/**
 * Same as the copy constructor.
 */
template<typename MetricType>
inline HRectBound<MetricType>& HRectBound<MetricType>::operator=(
    const HRectBound& other)
{
  if (dim != other.Dim())
  {
    // Reallocation is necessary.
    if (bounds)
      delete[] bounds;

    dim = other.Dim();
    bounds = new math::Range[dim];
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
template<typename MetricType>
inline HRectBound<MetricType>::HRectBound(HRectBound&& other) :
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
template<typename MetricType>
inline HRectBound<MetricType>::~HRectBound()
{
  if (bounds)
    delete[] bounds;
}

/**
 * Resets all dimensions to the empty set.
 */
template<typename MetricType>
inline void HRectBound<MetricType>::Clear()
{
  for (size_t i = 0; i < dim; i++)
    bounds[i] = math::Range();
  minWidth = 0;
}

/***
 * Calculates the centroid of the range, placing it into the given vector.
 *
 * @param centroid Vector which the centroid will be written to.
 */
template<typename MetricType>
inline void HRectBound<MetricType>::Center(arma::vec& center) const
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
template<typename MetricType>
inline double HRectBound<MetricType>::Volume() const
{
  double volume = 1.0;
  for (size_t i = 0; i < dim; ++i)
    volume *= (bounds[i].Hi() - bounds[i].Lo());

  return volume;
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename MetricType>
template<typename VecType>
inline double HRectBound<MetricType>::MinDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >* /* junk */) const
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
    // nonnegative half of the equation times two; then we raise to power Power.
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)),
        (double) MetricType::Power);
  }

  // Now take the Power'th root (but make sure our result is squared if it needs
  // to be); then cancel out the constant of 2 (which may have been squared now)
  // that was introduced earlier.  The compiler should optimize out the if
  // statement entirely.
  if (MetricType::TakeRoot)
    return pow(sum, 1.0 / (double) MetricType::Power) / 2.0;
  else
    return sum / pow(2.0, MetricType::Power);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename MetricType>
double HRectBound<MetricType>::MinDistance(const HRectBound& other) const
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
    sum += pow((lower + fabs(lower)) + (higher + fabs(higher)),
        (double) MetricType::Power);

    // Move bound pointers.
    mbound++;
    obound++;
  }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::TakeRoot)
    return pow(sum, 1.0 / (double) MetricType::Power) / 2.0;
  else
    return sum / pow(2.0, MetricType::Power);
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<typename MetricType>
template<typename VecType>
inline double HRectBound<MetricType>::MaxDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >* /* junk */) const
{
  double sum = 0;

  Log::Assert(point.n_elem == dim);

  for (size_t d = 0; d < dim; d++)
  {
    double v = std::max(fabs(point[d] - bounds[d].Lo()),
        fabs(bounds[d].Hi() - point[d]));
    sum += pow(v, (double) MetricType::Power);
  }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::TakeRoot)
    return pow(sum, 1.0 / (double) MetricType::Power);
  else
    return sum;
}

/**
 * Computes maximum distance.
 */
template<typename MetricType>
inline double HRectBound<MetricType>::MaxDistance(const HRectBound& other)
    const
{
  double sum = 0;

  Log::Assert(dim == other.dim);

  double v;
  for (size_t d = 0; d < dim; d++)
  {
    v = std::max(fabs(other.bounds[d].Hi() - bounds[d].Lo()),
        fabs(bounds[d].Hi() - other.bounds[d].Lo()));
    sum += pow(v, (double) MetricType::Power); // v is non-negative.
  }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::TakeRoot)
    return pow(sum, 1.0 / (double) MetricType::Power);
  else
    return sum;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<typename MetricType>
inline math::Range HRectBound<MetricType>::RangeDistance(
    const HRectBound& other) const
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

    loSum += pow(vLo, (double) MetricType::Power);
    hiSum += pow(vHi, (double) MetricType::Power);
  }

  if (MetricType::TakeRoot)
    return math::Range(pow(loSum, 1.0 / (double) MetricType::Power),
                       pow(hiSum, 1.0 / (double) MetricType::Power));
  else
    return math::Range(loSum, hiSum);
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<typename MetricType>
template<typename VecType>
inline math::Range HRectBound<MetricType>::RangeDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >* /* junk */) const
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

    loSum += pow(vLo, (double) MetricType::Power);
    hiSum += pow(vHi, (double) MetricType::Power);
  }

  if (MetricType::TakeRoot)
    return math::Range(pow(loSum, 1.0 / (double) MetricType::Power),
                       pow(hiSum, 1.0 / (double) MetricType::Power));
  else
    return math::Range(loSum, hiSum);
}

/**
 * Expands this region to include a new point.
 */
template<typename MetricType>
template<typename MatType>
inline HRectBound<MetricType>& HRectBound<MetricType>::operator|=(
    const MatType& data)
{
  Log::Assert(data.n_rows == dim);

  arma::vec mins(min(data, 1));
  arma::vec maxs(max(data, 1));

  minWidth = DBL_MAX;
  for (size_t i = 0; i < dim; i++)
  {
    bounds[i] |= math::Range(mins[i], maxs[i]);
    const double width = bounds[i].Width();
    if (width < minWidth)
      minWidth = width;
  }

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<typename MetricType>
inline HRectBound<MetricType>& HRectBound<MetricType>::operator|=(
    const HRectBound& other)
{
  assert(other.dim == dim);

  minWidth = DBL_MAX;
  for (size_t i = 0; i < dim; i++)
  {
    bounds[i] |= other.bounds[i];
    const double width = bounds[i].Width();
    if (width < minWidth)
      minWidth = width;
  }

  return *this;
}

/**
 * Determines if a point is within this bound.
 */
template<typename MetricType>
template<typename VecType>
inline bool HRectBound<MetricType>::Contains(const VecType& point) const
{
  for (size_t i = 0; i < point.n_elem; i++)
  {
    if (!bounds[i].Contains(point(i)))
      return false;
  }

  return true;
}

/**
 * Returns the diameter of the hyperrectangle (that is, the longest diagonal).
 */
template<typename MetricType>
inline double HRectBound<MetricType>::Diameter() const
{
  double d = 0;
  for (size_t i = 0; i < dim; ++i)
    d += std::pow(bounds[i].Hi() - bounds[i].Lo(),
        (double) MetricType::Power);

  if (MetricType::TakeRoot)
    return std::pow(d, 1.0 / (double) MetricType::Power);
  else
    return d;
}

//! Serialize the bound object.
template<typename MetricType>
template<typename Archive>
void HRectBound<MetricType>::Serialize(Archive& ar,
                                            const unsigned int /* version */)
{
  ar & data::CreateNVP(dim, "dim");

  // Allocate memory for the bounds, if necessary.
  if (Archive::is_loading::value)
  {
    if (bounds)
      delete[] bounds;
    bounds = new math::Range[dim];
  }

  ar & data::CreateArrayNVP(bounds, dim, "bounds");
  ar & data::CreateNVP(minWidth, "minWidth");
}

/**
 * Returns a string representation of this object.
 */
template<typename MetricType>
std::string HRectBound<MetricType>::ToString() const
{
  std::ostringstream convert;
  convert << "HRectBound [" << this << "]" << std::endl;
  convert << "  Power: " << MetricType::Power << std::endl;
  convert << "  TakeRoot: " << (MetricType::TakeRoot ? "true" : "false")
      << std::endl;
  convert << "  Dimensionality: " << dim << std::endl;
  convert << "  Bounds: " << std::endl;
  for (size_t i = 0; i < dim; ++i)
    convert << util::Indent(bounds[i].ToString()) << std::endl;
  convert << "  Minimum width: " << minWidth << std::endl;

  return convert.str();
}

} // namespace bound
} // namespace mlpack

#endif // __MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP
