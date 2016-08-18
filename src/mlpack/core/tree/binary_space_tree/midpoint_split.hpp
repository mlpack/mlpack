/**
 * @file midpoint_split.hpp
 * @author Yash Vadalia
 * @author Ryan Curtin
 *
 * Definition of MidpointSplit, a class that splits a binary space partitioning
 * tree node into two parts using the midpoint of the values in a certain
 * dimension.  The dimension to split on is the dimension with maximum variance.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_HPP

#include <mlpack/core.hpp>

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
