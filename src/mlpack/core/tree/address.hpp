
#ifndef MLPACK_CORE_TREE_ADDRESS_HPP
#define MLPACK_CORE_TREE_ADDRESS_HPP

namespace mlpack {

namespace bound {

namespace addr {

template<typename AddressType, typename VecType>
void PointToAddress(AddressType& address, const VecType& point)
{
  typedef typename VecType::elem_type VecElemType;
  typedef typename AddressType::elem_type AddressElemType;
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

  for (size_t i = 0; i < order; i++)
    for (size_t j = 0; j < point.n_elem; j++)
    {
      size_t bit = (i * point.n_elem + j) % order;
      size_t row = (i * point.n_elem + j) / order;

      address(row) |= (((result(j) >> (order - 1 - i)) & 1) <<
          (order - 1 - bit));
    }
}

template<typename AddressType, typename VecType>
void AddressToPoint(VecType& point, const AddressType& address)
{
  typedef typename VecType::elem_type VecElemType;
  typedef typename AddressType::elem_type AddressElemType;

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

    // Extract the mantissa.
    AddressElemType tmp = (AddressElemType) 1 << numMantBits;
    AddressElemType mantissa = rearrangedAddress(i) & (tmp - 1);

    VecElemType normalizedVal = (VecElemType) mantissa / tmp;

    if (!sgn)
      normalizedVal = -normalizedVal;

    // Extract the exponent
    tmp = (AddressElemType) 1 << numExpBits;
    AddressElemType e = (rearrangedAddress(i) >> numMantBits) & (tmp - 1);

    e += std::numeric_limits<VecElemType>::min_exponent;

    point(i) = std::ldexp(normalizedVal, e);
  }
}

template<typename AddressType1, typename AddressType2>
int CompareAddresses(const AddressType1& addr1, const AddressType2& addr2)
{
  static_assert(sizeof(typename AddressType1::elem_type) ==
      sizeof(typename AddressType2::elem_type), "We aren't able to compare "
      "adresses of distinct sizes");

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

template<typename AddressType1, typename AddressType2, typename AddressType3>
bool Contains(const AddressType1& address, const AddressType2& loBound,
                     const AddressType3& hiBound)
{
  static_assert(sizeof(typename AddressType1::elem_type) ==
      sizeof(typename AddressType2::elem_type), "We aren't able to compare "
      "adresses of distinct sizes");

  static_assert(sizeof(typename AddressType1::elem_type) ==
      sizeof(typename AddressType3::elem_type), "We aren't able to compare "
      "adresses of distinct sizes");

  assert(address.n_elem == loBound.n_elem);
  assert(address.n_elem == hiBound.n_elem);

  return ((CompareAddresses(loBound, address) <= 0) &&
          (CompareAddresses(hiBound, address) >= 0));
}

} // namespace addr

} // namespace bound

} // namespave mlpack

#endif // MLPACK_CORE_TREE_ADDRESS_HPP
