/**
 * @file cellbound_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the CellBound class. The class describes a bound that
 * consists of a number of hyperrectangles.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_CELLBOUND_IMPL_HPP
#define MLPACK_CORE_TREE_CELLBOUND_IMPL_HPP

#include <math.h>

// In case it has not been included yet.
#include "cellbound.hpp"

namespace mlpack {
namespace bound {

/**
 * Empty constructor.
 */
template<typename MetricType, typename ElemType>
inline CellBound<MetricType, ElemType>::CellBound() :
    dim(0),
    bounds(NULL),
    loBound(arma::Mat<ElemType>()),
    hiBound(arma::Mat<ElemType>()),
    numBounds(0),
    loAddress(arma::Col<AddressElemType>()),
    hiAddress(arma::Col<AddressElemType>()),
    minWidth(0)
{ /* Nothing to do. */ }

/**
 * Initializes to specified dimensionality with each dimension the empty
 * set.
 */
template<typename MetricType, typename ElemType>
inline CellBound<MetricType, ElemType>::CellBound(const size_t dimension) :
    dim(dimension),
    bounds(new math::RangeType<ElemType>[dim]),
    loBound(arma::Mat<ElemType>(dim, maxNumBounds)),
    hiBound(arma::Mat<ElemType>(dim, maxNumBounds)),
    numBounds(0),
    loAddress(dim),
    hiAddress(dim),
    minWidth(0)
{
  for (size_t k = 0; k < dim ; k++)
  {
    loAddress[k] = std::numeric_limits<AddressElemType>::max();
    hiAddress[k] = 0;
  }
}

/**
 * Copy constructor necessary to prevent memory leaks.
 */
template<typename MetricType, typename ElemType>
inline CellBound<MetricType, ElemType>::CellBound(
    const CellBound<MetricType, ElemType>& other) :
    dim(other.Dim()),
    bounds(new math::RangeType<ElemType>[dim]),
    loBound(other.loBound),
    hiBound(other.hiBound),
    numBounds(other.numBounds),
    loAddress(other.loAddress),
    hiAddress(other.hiAddress),
    minWidth(other.MinWidth())
{
  // Copy other bounds over.
  for (size_t i = 0; i < dim; i++)
    bounds[i] = other.bounds[i];
}

/**
 * Same as the copy constructor.
 */
template<typename MetricType, typename ElemType>
inline CellBound<MetricType, ElemType>& CellBound<MetricType, ElemType>::operator=(
    const CellBound<MetricType, ElemType>& other)
{
  if (dim != other.Dim())
  {
    // Reallocation is necessary.

    dim = other.Dim();
    bounds = new math::RangeType<ElemType>[dim];
  }

  loBound = other.loBound;
  hiBound = other.hiBound;
  numBounds = other.numBounds;
  loAddress = other.loAddress;
  hiAddress = other.hiAddress;

  // Now copy each of the bound values.
  for (size_t i = 0; i < dim; i++)
    bounds[i] = other.bounds[i];

  minWidth = other.MinWidth();

  return *this;
}

/**
 * Move constructor: take possession of another bound's information.
 */
template<typename MetricType, typename ElemType>
inline CellBound<MetricType, ElemType>::CellBound(
    CellBound<MetricType, ElemType>&& other) :
    dim(other.dim),
    bounds(other.bounds),
    loBound(std::move(other.loBound)),
    hiBound(std::move(other.hiBound)),
    numBounds(std::move(other.numBounds)),
    loAddress(std::move(other.loAddress)),
    hiAddress(std::move(other.hiAddress)),
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
inline CellBound<MetricType, ElemType>::~CellBound()
{
  if (bounds)
    delete[] bounds;
}

/**
 * Resets all dimensions to the empty set.
 */
template<typename MetricType, typename ElemType>
inline void CellBound<MetricType, ElemType>::Clear()
{
  for (size_t k = 0; k < dim; k++)
  {
    bounds[k] = math::RangeType<ElemType>();

    loAddress[k] = std::numeric_limits<AddressElemType>::max();
    hiAddress[k] = 0;
  }

  minWidth = 0;
}

/***
 * Calculates the centroid of the range, placing it into the given vector.
 *
 * @param centroid Vector which the centroid will be written to.
 */
template<typename MetricType, typename ElemType>
inline void CellBound<MetricType, ElemType>::Center(
    arma::Col<ElemType>& center) const
{
  // Set size correctly if necessary.
  if (!(center.n_elem == dim))
    center.set_size(dim);

  for (size_t i = 0; i < dim; i++)
    center(i) = bounds[i].Mid();
}

template<typename MetricType, typename ElemType>
template<typename MatType>
void CellBound<MetricType, ElemType>::AddBound(
    const arma::Col<ElemType>& loCorner,
    const arma::Col<ElemType>& hiCorner,
    const MatType& data)
{
  assert(numBounds < loBound.n_cols);
  assert(loBound.n_rows == dim);
  assert(loCorner.n_elem == dim);
  assert(hiCorner.n_elem == dim);

  for (size_t k = 0; k < dim; k++)
  {
    loBound(k, numBounds) = std::numeric_limits<ElemType>::max();
    hiBound(k, numBounds) = std::numeric_limits<ElemType>::lowest();
  }

  for (size_t i = 0; i < data.n_cols; i++)
  {
    size_t k = 0;
    // Check if the point is contained in the hyperrectangle.
    for (k = 0; k < dim; k++)
      if (data(k, i) < loCorner[k] || data(k, i) > hiCorner[k])
        break;

    if (k < dim)
      continue; // The point is not contained in the hyperrectangle.

    // Shrink the bound.
    for (k = 0; k < dim; k++)
    {
      loBound(k, numBounds) = std::min(loBound(k, numBounds), data(k, i));
      hiBound(k, numBounds) = std::max(hiBound(k, numBounds), data(k, i));
    }
  }

  for (size_t k = 0; k < dim; k++)
    if (loBound(k, numBounds) > hiBound(k, numBounds))
      return; // The hyperrectangle does not contain points.

  numBounds++;
}


template<typename MetricType, typename ElemType>
template<typename MatType>
void CellBound<MetricType, ElemType>::InitHighBound(size_t numEqualBits,
                                                    const MatType& data)
{
  arma::Col<AddressElemType> tmpHiAddress(hiAddress);
  arma::Col<AddressElemType> tmpLoAddress(hiAddress);
  arma::Col<ElemType> loCorner(tmpHiAddress.n_elem);
  arma::Col<ElemType> hiCorner(tmpHiAddress.n_elem);

  assert(tmpHiAddress.n_elem > 0);

  // We have to calculate the number of subrectangles since the maximum number
  // of hyperrectangles is restricted.
  size_t numCorners = 0;
  for (size_t pos = numEqualBits + 1; pos < order * tmpHiAddress.n_elem; pos++)
  {
    size_t row = pos / order;
    size_t bit = order - 1 - pos % order;

    // This hyperrectangle is not contained entirely in the bound.
    // So, the number of hyperrectangles should be increased.
    if (tmpHiAddress[row] & ((AddressElemType) 1 << bit))
      numCorners++;

    // We ran out of the limit of hyperrectangles. In that case we enlare
    // the last hyperrectangle.
    if (numCorners >= maxNumBounds / 2)
      tmpHiAddress[row] |= ((AddressElemType) 1 << bit);
  }

  size_t pos = order * tmpHiAddress.n_elem - 1;

  // Find the last hyperrectangle and add it to the bound.
  for ( ; pos > numEqualBits; pos--)
  {
    size_t row = pos / order;
    size_t bit = order - 1 - pos % order;

    // All last bits after pos of tmpHiAddress are equal to 1 and
    // All last bits of tmpLoAddress (after pos) are equal to 0.
    // Thus, tmpHiAddress corresponds to the high corner of the enlarged
    // rectangle and tmpLoAddress corresponds to the lower corner.
    if (!(tmpHiAddress[row] & ((AddressElemType) 1 << bit)))
    {
      addr::AddressToPoint(loCorner, tmpLoAddress);
      addr::AddressToPoint(hiCorner, tmpHiAddress);

      AddBound(loCorner, hiCorner, data);
      break;
    }
    // Nullify the bit that corresponds to this step.
    tmpLoAddress[row] &= ~((AddressElemType) 1 << bit);
  }

  // Add the enlarged rectangle if we have not done that.
  if (pos == numEqualBits)
  {
    addr::AddressToPoint(loCorner, tmpLoAddress);
    addr::AddressToPoint(hiCorner, tmpHiAddress);

    AddBound(loCorner, hiCorner, data);
  }

  for ( ; pos > numEqualBits; pos--)
  {
    size_t row = pos / order;
    size_t bit = order - 1 - pos % order;

    // The lower bound should correspond to this step.
    tmpLoAddress[row] &= ~((AddressElemType) 1 << bit);

    if (tmpHiAddress[row] & ((AddressElemType) 1 << bit))
    {
      // This hyperrectangle is contained entirely in the bound and do not
      // overlap with other hyperrectangles since loAddress is less than
      // tmpLoAddress and tmpHiAddress is less that the lower addresses
      // of hyperrectangles that we have added previously.
      tmpHiAddress[row] ^= (AddressElemType) 1 << bit;
      addr::AddressToPoint(loCorner, tmpLoAddress);
      addr::AddressToPoint(hiCorner, tmpHiAddress);

      AddBound(loCorner, hiCorner, data);
    }
    // The high bound should correspond to this step.
    tmpHiAddress[row] |= ((AddressElemType) 1 << bit);
  }
}

template<typename MetricType, typename ElemType>
template<typename MatType>
void CellBound<MetricType, ElemType>::InitLowerBound(size_t numEqualBits,
                                                     const MatType& data)
{
  arma::Col<AddressElemType> tmpHiAddress(loAddress);
  arma::Col<AddressElemType> tmpLoAddress(loAddress);
  arma::Col<ElemType> loCorner(tmpHiAddress.n_elem);
  arma::Col<ElemType> hiCorner(tmpHiAddress.n_elem);

  // We have to calculate the number of subrectangles since the maximum number
  // of hyperrectangles is restricted.
  size_t numCorners = 0;
  for (size_t pos = numEqualBits + 1; pos < order * tmpHiAddress.n_elem; pos++)
  {
    size_t row = pos / order;
    size_t bit = order - 1 - pos % order;

    // This hyperrectangle is not contained entirely in the bound.
    // So, the number of hyperrectangles should be increased.
    if (!(tmpLoAddress[row] & ((AddressElemType) 1 << bit)))
      numCorners++;

    // We ran out of the limit of hyperrectangles. In that case we enlare
    // the last hyperrectangle.
    if (numCorners >= maxNumBounds - numBounds)
      tmpLoAddress[row] &= ~((AddressElemType) 1 << bit);
  }

  size_t pos = order * tmpHiAddress.n_elem - 1;

  // Find the last hyperrectangle and add it to the bound.
  for ( ; pos > numEqualBits; pos--)
  {
    size_t row = pos / order;
    size_t bit = order - 1 - pos % order;

    // All last bits after pos of tmpHiAddress are equal to 1 and
    // All last bits of tmpLoAddress (after pos) are equal to 0.
    // Thus, tmpHiAddress corresponds to the high corner of the enlarged
    // rectangle and tmpLoAddress corresponds to the lower corner.
    if (tmpLoAddress[row] & ((AddressElemType) 1 << bit))
    {
      addr::AddressToPoint(loCorner, tmpLoAddress);
      addr::AddressToPoint(hiCorner, tmpHiAddress);

      AddBound(loCorner, hiCorner, data);
      break;
    }
    // Enlarge the hyperrectangle at this step since it is contained
    // entirely in the bound.
    tmpHiAddress[row] |= ((AddressElemType) 1 << bit);
  }

  // Add the enlarged rectangle if we have not done that.
  if (pos == numEqualBits)
  {
    addr::AddressToPoint(loCorner, tmpLoAddress);
    addr::AddressToPoint(hiCorner, tmpHiAddress);

    AddBound(loCorner, hiCorner, data);
  }

  for ( ; pos > numEqualBits; pos--)
  {
    size_t row = pos / order;
    size_t bit = order - 1 - pos % order;

    // The high bound should correspond to this step.
    tmpHiAddress[row] |= ((AddressElemType) 1 << bit);

    if (!(tmpLoAddress[row] & ((AddressElemType) 1 << bit)))
    {
      // This hyperrectangle is contained entirely in the bound and do not
      // overlap with other hyperrectangles since hiAddress is greater than
      // tmpHiAddress and tmpLoAddress is greater that the high addresses
      // of hyperrectangles that we have added previously.
      tmpLoAddress[row] ^= (AddressElemType) 1 << bit;

      addr::AddressToPoint(loCorner, tmpLoAddress);
      addr::AddressToPoint(hiCorner, tmpHiAddress);

      AddBound(loCorner, hiCorner, data);
    }

    // The lower bound should correspond to this step.
    tmpLoAddress[row] &= ~((AddressElemType) 1 << bit);
  }
}

template<typename MetricType, typename ElemType>
template<typename MatType>
void CellBound<MetricType, ElemType>::UpdateAddressBounds(const MatType& data)
{
  numBounds = 0;

  // Calculate the number of equal leading bits of the lower address and
  // the high address.
  size_t row = 0;
  for ( ; row < hiAddress.n_elem; row++)
    if (loAddress[row] != hiAddress[row])
      break;

  // If the high address is equal to the lower address.
  if (row == hiAddress.n_elem)
  {
    for (size_t i = 0; i < dim; i++)
    {
      loBound(i, 0) = bounds[i].Lo();
      hiBound(i, 0) = bounds[i].Hi();
    }
    numBounds = 1;

    return;
  }

  size_t bit = 0;
  for ( ; bit < order; bit++)
    if ((loAddress[row] & ((AddressElemType) 1 << (order - 1 - bit))) !=
        (hiAddress[row] & ((AddressElemType) 1 << (order - 1 - bit))))
      break;

  if ((row == hiAddress.n_elem - 1) && (bit == order - 1))
  {
    // If the addresses differ in the last bit.
    for (size_t i = 0; i < dim; i++)
    {
      loBound(i, 0) = bounds[i].Lo();
      hiBound(i, 0) = bounds[i].Hi();
    }

    numBounds = 1;

    return;
  }

  size_t numEqualBits = row * order + bit;
  InitHighBound(numEqualBits, data);
  InitLowerBound(numEqualBits, data);

  assert(numBounds <= maxNumBounds);

  if (numBounds == 0)
  {
    // I think this should never happen.
    for (size_t i = 0; i < dim; i++)
    {
      loBound(i, 0) = bounds[i].Lo();
      hiBound(i, 0) = bounds[i].Hi();
    }

    numBounds = 1;
  }
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename MetricType, typename ElemType>
template<typename VecType>
inline ElemType CellBound<MetricType, ElemType>::MinDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType>>* /* junk */) const
{
  Log::Assert(point.n_elem == dim);

  ElemType minSum = std::numeric_limits<ElemType>::max();

  ElemType lower, higher;

  for (size_t i = 0; i < numBounds; i++)
  {
    ElemType sum = 0;

    for (size_t d = 0; d < dim; d++)
    {
      lower = loBound(d, i) - point[d];
      higher = point[d] - hiBound(d, i);

      // Since only one of 'lower' or 'higher' is negative, if we add each's
      // absolute value to itself and then sum those two, our result is the
      // nonnegative half of the equation times two; then we raise to power Power.
      if (MetricType::Power == 1)
        sum += lower + std::fabs(lower) + higher + std::fabs(higher);
      else if (MetricType::Power == 2)
      {
        ElemType dist = lower + std::fabs(lower) + higher + std::fabs(higher);
        sum += dist * dist;
      }
      else
      {
        sum += pow((lower + fabs(lower)) + (higher + fabs(higher)),
            (ElemType) MetricType::Power);
      }

      if (sum >= minSum)
        break;
    }

    if (sum < minSum)
      minSum = sum;
  }

  // Now take the Power'th root (but make sure our result is squared if it needs
  // to be); then cancel out the constant of 2 (which may have been squared now)
  // that was introduced earlier.  The compiler should optimize out the if
  // statement entirely.
  if (MetricType::Power == 1)
    return minSum * 0.5;
  else if (MetricType::Power == 2)
  {
    if (MetricType::TakeRoot)
      return (ElemType) std::sqrt(minSum) * 0.5;
    else
      return minSum * 0.25;
  }
  else
  {
    if (MetricType::TakeRoot)
      return (ElemType) pow((double) minSum,
          1.0 / (double) MetricType::Power) / 2.0;
    else
      return minSum / pow(2.0, MetricType::Power);
  }
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename MetricType, typename ElemType>
ElemType CellBound<MetricType, ElemType>::MinDistance(const CellBound& other)
    const
{
  Log::Assert(dim == other.dim);

  ElemType minSum = std::numeric_limits<ElemType>::max();

  ElemType lower, higher;

  for (size_t i = 0; i < numBounds; i++)
    for (size_t j = 0; j < other.numBounds; j++)
    {
      ElemType sum = 0;
      for (size_t d = 0; d < dim; d++)
      {
        lower = other.loBound(d, j) - hiBound(d, i);
        higher = loBound(d, i) - other.hiBound(d, j);
        // We invoke the following:
        //   x + fabs(x) = max(x * 2, 0)
        //   (x * 2)^2 / 4 = x^2

        // The compiler should optimize out this if statement entirely.
        if (MetricType::Power == 1)
          sum += (lower + std::fabs(lower)) + (higher + std::fabs(higher));
        else if (MetricType::Power == 2)
        {
          ElemType dist = lower + std::fabs(lower) + higher + std::fabs(higher);
          sum += dist * dist;
        }
        else
        {
          sum += pow((lower + fabs(lower)) + (higher + fabs(higher)),
              (ElemType) MetricType::Power);
        }

        if (sum >= minSum)
          break;
      }

      if (sum < minSum)
        minSum = sum;
    }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::Power == 1)
    return minSum * 0.5;
  else if (MetricType::Power == 2)
  {
    if (MetricType::TakeRoot)
      return (ElemType) std::sqrt(minSum) * 0.5;
    else
      return minSum * 0.25;
  }
  else
  {
    if (MetricType::TakeRoot)
      return (ElemType) pow((double) minSum,
          1.0 / (double) MetricType::Power) / 2.0;
    else
      return minSum / pow(2.0, MetricType::Power);
  }
}

/**
 * Calculates maximum bound-to-point squared distance.
 */
template<typename MetricType, typename ElemType>
template<typename VecType>
inline ElemType CellBound<MetricType, ElemType>::MaxDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >* /* junk */) const
{
  ElemType maxSum = std::numeric_limits<ElemType>::lowest();

  Log::Assert(point.n_elem == dim);

  for (size_t i = 0; i < numBounds; i++)
  {
    ElemType sum = 0;
    for (size_t d = 0; d < dim; d++)
    {
      ElemType v = std::max(fabs(point[d] - loBound(d, i)),
          fabs(hiBound(d, i) - point[d]));

      if (MetricType::Power == 1)
        sum += v; // v is non-negative.
      else if (MetricType::Power == 2)
        sum += v * v;
      else
        sum += std::pow(v, (ElemType) MetricType::Power);
    }

    if (sum > maxSum)
      maxSum = sum;
  }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return maxSum;
    else if (MetricType::Power == 2)
      return (ElemType) std::sqrt(maxSum);
    else
      return (ElemType) pow((double) maxSum, 1.0 / (double) MetricType::Power);
  }
  else
    return maxSum;
}

/**
 * Computes maximum distance.
 */
template<typename MetricType, typename ElemType>
inline ElemType CellBound<MetricType, ElemType>::MaxDistance(
    const CellBound& other)
    const
{
  ElemType maxSum = std::numeric_limits<ElemType>::lowest();

  Log::Assert(dim == other.dim);

  ElemType v;
  for (size_t i = 0; i < numBounds; i++)
    for (size_t j = 0; j < other.numBounds; j++)
    {
      ElemType sum = 0;
      for (size_t d = 0; d < dim; d++)
      {
        v = std::max(fabs(other.hiBound(d, j) - loBound(d, i)),
            fabs(hiBound(d, i) - other.loBound(d, j)));

        // The compiler should optimize out this if statement entirely.
        if (MetricType::Power == 1)
          sum += v; // v is non-negative.
        else if (MetricType::Power == 2)
          sum += v * v;
        else
          sum += std::pow(v, (ElemType) MetricType::Power);
      }

      if (sum > maxSum)
        maxSum = sum;
    }

  // The compiler should optimize out this if statement entirely.
  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return maxSum;
    else if (MetricType::Power == 2)
      return (ElemType) std::sqrt(maxSum);
    else
      return (ElemType) pow((double) maxSum, 1.0 / (double) MetricType::Power);
  }
  else
    return maxSum;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 */
template<typename MetricType, typename ElemType>
inline math::RangeType<ElemType>
CellBound<MetricType, ElemType>::RangeDistance(
    const CellBound& other) const
{
  ElemType minLoSum = std::numeric_limits<ElemType>::max();
  ElemType maxHiSum = std::numeric_limits<ElemType>::lowest();

  Log::Assert(dim == other.dim);

  ElemType v1, v2, vLo, vHi;

  for (size_t i = 0; i < numBounds; i++)
    for (size_t j = 0; j < other.numBounds; j++)
    {
      ElemType loSum = 0;
      ElemType hiSum = 0;
      for (size_t d = 0; d < dim; d++)
      {
        v1 = other.loBound(d, j) - hiBound(d, i);
        v2 = loBound(d, i) - other.hiBound(d, j);
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

      if (loSum < minLoSum)
        minLoSum = loSum;
      if (hiSum > maxHiSum)
        maxHiSum = hiSum;
    }

  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return math::RangeType<ElemType>(minLoSum, maxHiSum);
    else if (MetricType::Power == 2)
      return math::RangeType<ElemType>((ElemType) std::sqrt(minLoSum),
                                       (ElemType) std::sqrt(maxHiSum));
    else
    {
      return math::RangeType<ElemType>(
          (ElemType) pow((double) minLoSum, 1.0 / (double) MetricType::Power),
          (ElemType) pow((double) maxHiSum, 1.0 / (double) MetricType::Power));
    }
  }
  else
    return math::RangeType<ElemType>(minLoSum, maxHiSum);
}

/**
 * Calculates minimum and maximum bound-to-point squared distance.
 */
template<typename MetricType, typename ElemType>
template<typename VecType>
inline math::RangeType<ElemType>
CellBound<MetricType, ElemType>::RangeDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType>>* /* junk */) const
{
  ElemType minLoSum = std::numeric_limits<ElemType>::max();
  ElemType maxHiSum = std::numeric_limits<ElemType>::lowest();

  Log::Assert(point.n_elem == dim);

  ElemType v1, v2, vLo, vHi;
  for (size_t i = 0; i < numBounds; i++)
  {
    ElemType loSum = 0;
    ElemType hiSum = 0;
    for (size_t d = 0; d < dim; d++)
    {
      v1 = loBound(d, i) - point[d]; // Negative if point[d] > lo.
      v2 = point[d] - hiBound(d, i); // Negative if point[d] < hi.
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
    if (loSum < minLoSum)
      minLoSum = loSum;
    if (hiSum > maxHiSum)
      maxHiSum = hiSum;
  }

  if (MetricType::TakeRoot)
  {
    if (MetricType::Power == 1)
      return math::RangeType<ElemType>(minLoSum, maxHiSum);
    else if (MetricType::Power == 2)
      return math::RangeType<ElemType>((ElemType) std::sqrt(minLoSum),
                                       (ElemType) std::sqrt(maxHiSum));
    else
    {
      return math::RangeType<ElemType>(
          (ElemType) pow((double) minLoSum, 1.0 / (double) MetricType::Power),
          (ElemType) pow((double) maxHiSum, 1.0 / (double) MetricType::Power));
    }
  }
  else
    return math::RangeType<ElemType>(minLoSum, maxHiSum);
}

/**
 * Expands this region to include a new point.
 */
template<typename MetricType, typename ElemType>
template<typename MatType>
inline CellBound<MetricType, ElemType>& CellBound<MetricType, ElemType>::operator|=(
    const MatType& data)
{
  Log::Assert(data.n_rows == dim);

  arma::Col<ElemType> mins(arma::min(data, 1));
  arma::Col<ElemType> maxs(arma::max(data, 1));

  minWidth = std::numeric_limits<ElemType>::max();
  for (size_t i = 0; i < dim; i++)
  {
    bounds[i] |= math::RangeType<ElemType>(mins[i], maxs[i]);
    const ElemType width = bounds[i].Width();
    if (width < minWidth)
      minWidth = width;

    loBound(i, 0) = bounds[i].Lo();
    hiBound(i, 0) = bounds[i].Hi();
  }

  numBounds = 1;

  return *this;
}

/**
 * Expands this region to encompass another bound.
 */
template<typename MetricType, typename ElemType>
inline CellBound<MetricType, ElemType>& CellBound<MetricType, ElemType>::operator|=(
    const CellBound& other)
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

  if (addr::CompareAddresses(other.loAddress, loAddress) < 0)
    loAddress = other.loAddress;

  if (addr::CompareAddresses(other.hiAddress, hiAddress) > 0)
    hiAddress = other.hiAddress;

  if (loAddress[0] > hiAddress[0])
  {
    for (size_t i = 0; i < dim; i++)
    {
      loBound(i, 0) = bounds[i].Lo();
      hiBound(i, 0) = bounds[i].Hi();
    }
    numBounds = 1;
  }
  return *this;
}

/**
 * Determines if a point is within this bound.
 */
template<typename MetricType, typename ElemType>
template<typename VecType>
inline bool CellBound<MetricType, ElemType>::Contains(const VecType& point) const
{
  for (size_t i = 0; i < point.n_elem; i++)
  {
    if (!bounds[i].Contains(point(i)))
      return false;
  }

  if (loAddress[0] > hiAddress[0])
    return true;

  arma::Col<AddressElemType> address(dim);

  addr::PointToAddress(address, point);

  return addr::Contains(address, loAddress, hiAddress);
}


/**
 * Returns the diameter of the hyperrectangle (that is, the longest diagonal).
 */
template<typename MetricType, typename ElemType>
inline ElemType CellBound<MetricType, ElemType>::Diameter() const
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
void CellBound<MetricType, ElemType>::Serialize(Archive& ar,
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
  ar & data::CreateNVP(loBound, "loBound");
  ar & data::CreateNVP(hiBound, "hiBound");
  ar & data::CreateNVP(numBounds, "numBounds");
  ar & data::CreateNVP(loAddress, "loAddress");
  ar & data::CreateNVP(hiAddress, "hiAddress");
}

} // namespace bound
} // namespace mlpack

#endif // MLPACK_CORE_TREE_HRECTBOUND_IMPL_HPP

