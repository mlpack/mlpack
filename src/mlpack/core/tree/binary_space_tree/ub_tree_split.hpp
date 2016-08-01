/**
 * @file ub_tree_split.hpp
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_UB_TREE_SPLIT_HPP

#include <mlpack/core.hpp>
#include "../address.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename BoundType, typename MatType = arma::mat>
class UBTreeSplit
{
 public:
  typedef typename std::conditional<sizeof(typename MatType::elem_type) * CHAR_BIT <= 32,
                                    uint32_t,
                                    uint64_t>::type AddressElemType;

  bool SplitNode(BoundType& bound,
                 MatType& data,
                 const size_t begin,
                 const size_t count,
                 size_t& splitCol);

  bool SplitNode(BoundType& bound,
                 MatType& data,
                 const size_t begin,
                 const size_t count,
                 size_t& splitCol,
                 std::vector<size_t>& oldFromNew);

 private:
//  arma::Mat<AddressElemType> addresses;
  std::vector<std::pair<arma::Col<AddressElemType>, size_t>> addresses;

  template<typename VecType>
  arma::Col<AddressElemType> CalculateAddress(const VecType& point);

  void InitializeAddresses(const MatType& data);

  void PerformSplit(MatType& data,
                       const size_t count);

  void PerformSplit(MatType& data,
                       const size_t count,
                       std::vector<size_t>& oldFromNew);

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
