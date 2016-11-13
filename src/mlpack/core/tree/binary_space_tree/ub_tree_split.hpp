/**
 * @file ub_tree_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of UBTreeSplit, a class that splits the space according
 * to the median address of points contained in the node.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_HPP

#include <mlpack/core.hpp>
#include "../address.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * Split a node into two parts according to the median address of points
 * contained in the node. The class reorders the dataset such that points
 * with lower addresses belong to the left subtree and points with high
 * addresses belong to the right subtree.
 */
template<typename BoundType, typename MatType = arma::mat>
class UBTreeSplit
{
 public:
  //! The type of an address element.
  typedef typename std::conditional<sizeof(typename MatType::elem_type) * CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type AddressElemType;

  //! An information about the partition.
  struct SplitInfo
  {
    //! This vector contains addresses of all points in the dataset.
    std::vector<std::pair<arma::Col<AddressElemType>, size_t>>* addresses;
  };

  /**
   * Split the node according to the median address of points contained in the
   * node.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitInfo An information about the split (not used here).
   */
  bool SplitNode(BoundType& bound,
                 MatType& data,
                 const size_t begin,
                 const size_t count,
                 SplitInfo&  splitInfo);

  /**
   * Rearrange the dataset according to the addresses.
   *
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitInfo The information about the split.
   */
  static size_t PerformSplit(MatType& data,
                             const size_t begin,
                             const size_t count,
                             const SplitInfo& splitInfo);

  /**
   * Rearrange the dataset according to the addresses and return the list
   * of changed indices.
   *
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitInfo The information about the split.
   * @param oldFromNew Vector which will be filled with the old positions for
   *    each new point.
   */
  static size_t PerformSplit(MatType& data,
                             const size_t begin,
                             const size_t count,
                             const SplitInfo& splitInfo,
                             std::vector<size_t>& oldFromNew);

 private:
  //! This vector contains addresses of all points in the dataset.
  std::vector<std::pair<arma::Col<AddressElemType>, size_t>> addresses;

  /**
   * Calculate addresses for all points in the dataset.
   *
   * @param data The dataset used by the binary space tree.
   */
  void InitializeAddresses(const MatType& data);

  //! A comparator for sorting addresses.
  static bool ComparePair(
      const std::pair<arma::Col<AddressElemType>, size_t>& p1,
      const std::pair<arma::Col<AddressElemType>, size_t>& p2)
  {
    return bound::addr::CompareAddresses(p1.first, p2.first) < 0;
  }
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "ub_tree_split_impl.hpp"

#endif
