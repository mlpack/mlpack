/**
 * @file midpoint_split.hpp
 * @author Yash Vadalia
 * @author Ryan Curtin
 *
 * Definition of MidpointSplit, a class that splits a binary space partitioning
 * tree node into two parts using the midpoint of the values in a certain
 * dimension.  The dimension to split on is the dimension with maximum variance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/perform_split.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A binary space partitioning tree node is split into its left and right child.
 * The split is done in the dimension that has the maximum width. The points are
 * divided into two parts based on the midpoint in this dimension.
 */
template<typename BoundType, typename MatType = arma::mat>
class MidpointSplit
{
 public:
  //! A struct that contains an information about the split.
  struct SplitInfo
  {
    //! The dimension to split the node on.
    size_t splitDimension;
    //! The split in dimension splitDimension is based on this value.
    double splitVal;
  };
  /**
   * Find the partition of the node. This method fills up the dimension that
   * will be used to split the node and the value according which the split
   * will be performed.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitInfo An information about the split. This information contains
   *    the dimension and the value.
   */
  static bool SplitNode(const BoundType& bound,
                        MatType& data,
                        const size_t begin,
                        const size_t count,
                        SplitInfo& splitInfo);

  /**
   * Perform the split process according to the information about the
   * split. This will order the dataset such that points that belong to the left
   * subtree are on the left of the split column, and points from the right
   * subtree are on the right side of the split column.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitInfo The information about the split.
   */
  static size_t PerformSplit(MatType& data,
                             const size_t begin,
                             const size_t count,
                             const SplitInfo& splitInfo)
  {
    return split::PerformSplit<MatType, MidpointSplit>(data, begin, count,
        splitInfo);
  }

  /**
   * Perform the split process according to the information about the split and
   * return the list of changed indices. This will order the dataset such that
   * points that belong to the left subtree are on the left of the split column,
   * and points from the right subtree are on the right side of the split
   * column.
   *
   * @param bound The bound used for this node.
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
                             std::vector<size_t>& oldFromNew)
  {
    return split::PerformSplit<MatType, MidpointSplit>(data, begin, count,
        splitInfo, oldFromNew);
  }

  /**
   * Indicates that a point should be assigned to the left subtree.
   *
   * @param point The point that is being assigned.
   * @param splitInfo An information about the split.
   */
  template<typename VecType>
  static bool AssignToLeftNode(const VecType& point,
                               const SplitInfo& splitInfo)
  {
    return point[splitInfo.splitDimension] < splitInfo.splitVal;
  }
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "midpoint_split_impl.hpp"

#endif
