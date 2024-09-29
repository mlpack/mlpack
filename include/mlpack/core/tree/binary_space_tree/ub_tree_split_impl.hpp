/**
 * @file core/tree/binary_space_tree/ub_tree_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of UBTreeSplit, a class that splits a node according
 * to the median address of points contained in the node.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_IMPL_HPP

#include "ub_tree_split.hpp"
#include <mlpack/core/tree/bounds.hpp>

namespace mlpack {

template<typename BoundType, typename MatType>
bool UBTreeSplit<BoundType, MatType>::SplitNode(BoundType& bound,
                                                MatType& data,
                                                const size_t begin,
                                                const size_t count,
                                                SplitInfo& splitInfo)
{
  constexpr size_t order = sizeof(AddressElemType) * CHAR_BIT;
  if (begin == 0 && count == data.n_cols)
  {
    // Calculate all addresses.
    InitializeAddresses(data);

    // Probably this is not a good idea. Maybe it is better to get
    // a number of distinct samples and find the median.
    std::sort(addresses.begin(), addresses.end(), ComparePair);

    // Save the vector in order to rearrange the dataset later.
    splitInfo.addresses = &addresses;
  }
  else
  {
    // We have already rearranged the dataset.
    splitInfo.addresses = NULL;
  }

  // The bound shouldn't contain too many subrectangles.
  // In order to minimize the number of hyperrectangles we set last bits
  // of the last address in the node to 1 and last bits of the first  address
  // in the next node to zero in such a way that the ordering is not
  // disturbed.
  if (begin + count < data.n_cols)
  {
    // Omit leading equal bits.
    size_t row = 0;
    arma::Col<AddressElemType>& lo = addresses[begin + count - 1].first;
    const arma::Col<AddressElemType>& hi = addresses[begin + count].first;

    for (; row < data.n_rows; row++)
      if (lo[row] != hi[row])
        break;

    size_t bit = 0;

    for (; bit < order; bit++)
      if ((lo[row] & ((AddressElemType) 1 << (order - 1 - bit))) !=
          (hi[row] & ((AddressElemType) 1 << (order - 1 - bit))))
        break;

    bit++;

    // Replace insignificant bits.
    if (bit == order)
    {
      bit = 0;
      row++;
    }
    else
    {
      for (; bit < order; bit++)
        lo[row] |= ((AddressElemType) 1 << (order - 1 - bit));
      row++;
    }

    for (; row < data.n_rows; row++)
      for (; bit < order; bit++)
        lo[row] |= ((AddressElemType) 1 << (order - 1 - bit));
  }

  // The bound shouldn't contain too many subrectangles.
  // In order to minimize the number of hyperrectangles we set last bits
  // of the first address in the next node to 0 and last bits of the last
  // address in the previous node to 1 in such a way that the ordering is not
  // disturbed.
  if (begin > 0)
  {
    // Omit leading equal bits.
    size_t row = 0;
    const arma::Col<AddressElemType>& lo = addresses[begin - 1].first;
    arma::Col<AddressElemType>& hi = addresses[begin].first;

    for (; row < data.n_rows; row++)
      if (lo[row] != hi[row])
        break;

    size_t bit = 0;

    for (; bit < order; bit++)
      if ((lo[row] & ((AddressElemType) 1 << (order - 1 - bit))) !=
          (hi[row] & ((AddressElemType) 1 << (order - 1 - bit))))
        break;

    bit++;

    // Replace insignificant bits.
    if (bit == order)
    {
      bit = 0;
      row++;
    }
    else
    {
      for (; bit < order; bit++)
        hi[row] &= ~((AddressElemType) 1 << (order - 1 - bit));
      row++;
    }

    for (; row < data.n_rows; row++)
      for (; bit < order; bit++)
        hi[row] &= ~((AddressElemType) 1 << (order - 1 - bit));
  }

  // Set the minimum and the maximum addresses.
  for (size_t k = 0; k < bound.Dim(); ++k)
  {
    bound.LoAddress()[k] = addresses[begin].first[k];
    bound.HiAddress()[k] = addresses[begin + count - 1].first[k];
  }
  bound.UpdateAddressBounds(data.cols(begin, begin + count - 1));

  return true;
}

template<typename BoundType, typename MatType>
void UBTreeSplit<BoundType, MatType>::InitializeAddresses(const MatType& data)
{
  addresses.resize(data.n_cols);

  // Calculate all addresses.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    addresses[i].first.zeros(data.n_rows);
    PointToAddress(addresses[i].first, data.col(i));
    addresses[i].second = i;
  }
}

template<typename BoundType, typename MatType>
size_t UBTreeSplit<BoundType, MatType>::PerformSplit(
    MatType& data,
    const size_t begin,
    const size_t count,
    const SplitInfo& splitInfo)
{
  // For the first time we have to rearrange the dataset.
  if (splitInfo.addresses)
  {
    std::vector<size_t> newFromOld(data.n_cols);
    std::vector<size_t> oldFromNew(data.n_cols);

    for (size_t i = 0; i < splitInfo.addresses->size(); ++i)
    {
      newFromOld[i] = i;
      oldFromNew[i] = i;
    }

    for (size_t i = 0; i < splitInfo.addresses->size(); ++i)
    {
      size_t index = (*splitInfo.addresses)[i].second;
      size_t oldI = oldFromNew[i];
      size_t newIndex = newFromOld[index];

      data.swap_cols(i, newFromOld[index]);

      size_t tmp = newFromOld[index];
      newFromOld[index] = i;
      newFromOld[oldI] = tmp;

      tmp = oldFromNew[i];
      oldFromNew[i] = oldFromNew[newIndex];
      oldFromNew[newIndex] = tmp;
    }
  }

  // Since the dataset is sorted we can easily obtain the split column.
  return begin + count / 2;
}

template<typename BoundType, typename MatType>
size_t UBTreeSplit<BoundType, MatType>::PerformSplit(
    MatType& data,
    const size_t begin,
    const size_t count,
    const SplitInfo& splitInfo,
    std::vector<size_t>& oldFromNew)
{
  // For the first time we have to rearrange the dataset.
  if (splitInfo.addresses)
  {
    std::vector<size_t> newFromOld(data.n_cols);

    for (size_t i = 0; i < splitInfo.addresses->size(); ++i)
      newFromOld[i] = i;

    for (size_t i = 0; i < splitInfo.addresses->size(); ++i)
    {
      size_t index = (*splitInfo.addresses)[i].second;
      size_t oldI = oldFromNew[i];
      size_t newIndex = newFromOld[index];

      data.swap_cols(i, newFromOld[index]);

      size_t tmp = newFromOld[index];
      newFromOld[index] = i;
      newFromOld[oldI] = tmp;

      tmp = oldFromNew[i];
      oldFromNew[i] = oldFromNew[newIndex];
      oldFromNew[newIndex] = tmp;
    }
  }

  // Since the dataset is sorted we can easily obtain the split column.
  return begin + count / 2;
}

} // namespace mlpack

#endif
