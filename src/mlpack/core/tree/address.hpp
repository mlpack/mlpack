/**
 * @file address.hpp
 * @author Mikhail Lozhnikov
 *
 * This file contains a series of functions for translating points to addresses
 * and back and functions for comparing addresses.
 *
 * The notion of addresses is described in the following paper.
 * @code
 * @inproceedings{bayer1997,
 *   author = {Bayer, Rudolf},
 *   title = {The Universal B-Tree for Multidimensional Indexing: General
 *       Concepts},
 *   booktitle = {Proceedings of the International Conference on Worldwide
 *       Computing and Its Applications},
 *   series = {WWCA '97},
 *   year = {1997},
 *   isbn = {3-540-63343-X},
 *   pages = {198--209},
 *   numpages = {12},
 *   publisher = {Springer-Verlag},
 *   address = {London, UK, UK},
 * }
 * @endcode
*/
#ifndef MLPACK_CORE_TREE_ADDRESS_HPP
#define MLPACK_CORE_TREE_ADDRESS_HPP

namespace mlpack {
namespace bound {
namespace addr {

/**
 * Calculate the address of a point. Be careful, the point and the address
 * variables should be equal-sized and the type of the address should correspond
 * to the type of the vector.
 *
 * The function maps each floating point coordinate to an equal-sized unsigned
 * integer datatype in such a way that the transform preserves the ordering
 * (i.e. lower floating point values correspond to lower integers). Thus,
 * the mapping saves the exponent and the mantissa of each floating point value
 * consequently, furthermore the exponent is stored before the mantissa. In the
 * case of negative numbers the resulting integer value should be inverted.
 * In the multi-dimensional case, after we transform the representation, we
 * have to interleave the bits of the new representation across all the elements
 * in the address vector.
 *
 * @param address The resulting address.
 * @param point The point that is being translated to the address.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
template<typename AddressType, typename VecType>
void PointToAddress(AddressType& address, const VecType& point)
{
  typedef typename VecType::elem_type VecElemType;
  // Check that the arguments are compatible.
  typedef typename std::conditional<sizeof(VecElemType) * CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type AddressElemType;

  static_assert(std::is_same<typename AddressType::elem_type,
      AddressElemType>::value == true, "The vector element type does not "
      "correspond to the address element type.");
  arma::Col<AddressElemType> result(point.n_elem);

  constexpr size_t order = sizeof(AddressElemType) * CHAR_BIT;
  // Calculate the number of bits for the exponent.
  const int numExpBits = std::ceil(std::log2(
      std::numeric_limits<VecElemType>::max_exponent -
      std::numeric_limits<VecElemType>::min_exponent + 1.0));

  // Calculate the number of bits for the mantissa.
  const int numMantBits = order - numExpBits - 1;

  assert(point.n_elem == address.n_elem);
  assert(address.n_elem > 0);

  for (size_t i = 0; i < point.n_elem; i++)
  {
    int e;
    VecElemType normalizedVal = std::frexp(point(i),&e);
    bool sgn = std::signbit(normalizedVal);

    if (point(i) == 0)
      e = std::numeric_limits<VecElemType>::min_exponent;

    if (sgn)
      normalizedVal = -normalizedVal;

    if (e < std::numeric_limits<VecElemType>::min_exponent)
    {
      AddressElemType tmp = (AddressElemType) 1 <<
          (std::numeric_limits<VecElemType>::min_exponent - e);

      e = std::numeric_limits<VecElemType>::min_exponent;
      normalizedVal /= tmp;
    }

    // Extract the mantissa.
    AddressElemType tmp = (AddressElemType) 1 << numMantBits;
    result(i) = std::floor(normalizedVal * tmp);

    // Add the exponent.
    assert(result(i) < ((AddressElemType) 1 << numMantBits));
    result(i) |= ((AddressElemType)
        (e - std::numeric_limits<VecElemType>::min_exponent)) << numMantBits;

    assert(result(i) < ((AddressElemType) 1 << (order - 1)) - 1);

    // Negative values should be inverted.
    if (sgn)
    {
      result(i) = ((AddressElemType) 1 << (order - 1)) - 1 - result(i);
      assert((result(i) >> (order - 1)) == 0);
    }
    else
    {
      result(i) |= (AddressElemType) 1 << (order - 1);
      assert((result(i) >> (order - 1)) == 1);
    }
  }

  address.zeros(point.n_elem);

  // Interleave the bits of the new representation across all the elements
  // in the address vector.
  for (size_t i = 0; i < order; i++)
    for (size_t j = 0; j < point.n_elem; j++)
    {
      size_t bit = (i * point.n_elem + j) % order;
      size_t row = (i * point.n_elem + j) / order;

      address(row) |= (((result(j) >> (order - 1 - i)) & 1) <<
          (order - 1 - bit));
    }
}

/**
 * Translate the address to the point. Be careful, the point and the address
 * variables should be equal-sized and the type of the address should correspond
 * to the type of the vector.
 *
 * The function makes the backward transform to the function above.
 *
 * @param address An address to translate.
 * @param point The point that corresponds to the address.
 */
template<typename AddressType, typename VecType>
void AddressToPoint(VecType& point, const AddressType& address)
{
  typedef typename VecType::elem_type VecElemType;
  // Check that the arguments are compatible.
  typedef typename std::conditional<sizeof(VecElemType) * CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type AddressElemType;

  static_assert(std::is_same<typename AddressType::elem_type,
      AddressElemType>::value == true, "The vector element type does not "
      "correspond to the address element type.");

  constexpr size_t order = sizeof(AddressElemType) * CHAR_BIT;
  // Calculate the number of bits for the exponent.
  const int numExpBits = std::ceil(std::log2(
      std::numeric_limits<VecElemType>::max_exponent -
      std::numeric_limits<VecElemType>::min_exponent + 1.0));

  assert(point.n_elem == address.n_elem);
  assert(address.n_elem > 0);

  arma::Col<AddressElemType> rearrangedAddress(address.n_elem,
      arma::fill::zeros);
  // Calculate the number of bits for the mantissa.
  const int numMantBits = order - numExpBits - 1;

  for (size_t i = 0; i < order; i++)
    for (size_t j = 0; j < address.n_elem; j++)
    {
      size_t bit = (i * address.n_elem + j) % order;
      size_t row = (i * address.n_elem + j) / order;

      rearrangedAddress(j) |= (((address(row) >> (order - 1 - bit)) & 1) <<
          (order - 1 - i));
    }

  for (size_t i = 0; i < rearrangedAddress.n_elem; i++)
  {
    bool sgn = rearrangedAddress(i) & ((AddressElemType) 1 << (order - 1));

    if (!sgn)
    {
      rearrangedAddress(i) = ((AddressElemType) 1 << (order - 1)) - 1 -
          rearrangedAddress(i);
    }

    // Extract the mantissa.
    AddressElemType tmp = (AddressElemType) 1 << numMantBits;
    AddressElemType mantissa = rearrangedAddress(i) & (tmp - 1);
    if (mantissa == 0)
      mantissa = 1;

    VecElemType normalizedVal = (VecElemType) mantissa / tmp;

    if (!sgn)
      normalizedVal = -normalizedVal;

    // Extract the exponent
    tmp = (AddressElemType) 1 << numExpBits;
    AddressElemType e = (rearrangedAddress(i) >> numMantBits) & (tmp - 1);

    e += std::numeric_limits<VecElemType>::min_exponent;

    point(i) = std::ldexp(normalizedVal, e);
    if (std::isinf(point(i)))
    {
      if (point(i) > 0)
        point(i) = std::numeric_limits<VecElemType>::max();
      else
        point(i) = std::numeric_limits<VecElemType>::lowest();
    }
  }
}

/**
 * Compare two addresses. The function returns 1 if the first address is greater
 * than the second one, -1 if the first address is less than the second one,
 * otherwise the function returns 0.
 */
template<typename AddressType1, typename AddressType2>
int CompareAddresses(const AddressType1& addr1, const AddressType2& addr2)
{
  static_assert(std::is_same<typename AddressType1::elem_type,
      typename AddressType2::elem_type>::value == true, "Can't compare "
      "addresses of distinct types");

  assert(addr1.n_elem == addr2.n_elem);

  for (size_t i = 0; i < addr1.n_elem; i++)
  {
    if (addr1[i] < addr2[i])
      return -1;
    else if (addr2[i] < addr1[i])
      return 1;
  }

  return 0;
}

/**
 * Returns true if an address is contained between two other addresses.
 */
template<typename AddressType1, typename AddressType2, typename AddressType3>
bool Contains(const AddressType1& address, const AddressType2& loBound,
                     const AddressType3& hiBound)
{
  return ((CompareAddresses(loBound, address) <= 0) &&
          (CompareAddresses(hiBound, address) >= 0));
}

} // namespace addr
} // namespace bound
} // namespave mlpack

#endif // MLPACK_CORE_TREE_ADDRESS_HPP
