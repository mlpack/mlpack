/**
 * @file mean_split.hpp
 * @author Yash Vadalia
 * @author Ryan Curtin
 *
 * Definition of MeanSplit, a class that splits a binary space partitioning tree
 * node into two parts using the mean of the values in a certain dimension.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_MEAN_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_MEAN_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A binary space partitioning tree node is split into its left and right child.
 * The split is done in the dimension that has the maximum width. The points are
 * divided into two parts based on the mean in this dimension.
 */
template<typename BoundType, typename MatType = arma::mat>
class MeanSplit
{
 public:
  /**
   * Split the node according to the mean value in the dimension with maximum
   * width.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitDimension This will be filled with the dimension the node is to
   *    be split on.
   * @param splitCol The index at which the dataset is divided into two parts
   *    after the rearrangement.
   */
  static bool SplitNode(const BoundType& bound,
                        MatType& data,
                        const size_t begin,
                        const size_t count,
                        size_t& splitCol);

  /**
   * Split the node according to the mean value in the dimension with maximum
   * width and return a list of changed indices.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitDimension This will be filled with the dimension the node is
   *    to be split on.
   * @param splitCol The index at which the dataset is divided into two parts
   *    after the rearrangement.
   * @param oldFromNew Vector which will be filled with the old positions for
   *    each new point.
   */
  static bool SplitNode(const BoundType& bound,
                        MatType& data,
                        const size_t begin,
                        const size_t count,
                        size_t& splitCol,
                        std::vector<size_t>& oldFromNew);

 private:
  /**
   * Reorder the dataset into two parts such that they lie on either side of
   * splitCol.
   *
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitDimension The dimension to split the node on.
   * @param splitVal The split in dimension splitDimension is based on this
   *    value.
   */
  static size_t PerformSplit(MatType& data,
                             const size_t begin,
                             const size_t count,
                             const size_t splitDimension,
                             const double splitVal);

  /**
   * Reorder the dataset into two parts such that they lie on either side of
   * splitCol. Also returns a list of changed indices.
   *
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitDimension The dimension to split the node on.
   * @param splitVal The split in dimension splitDimension is based on this
   *    value.
   * @param oldFromNew Vector which will be filled with the old positions for
   *    each new point.
   */
  static size_t PerformSplit(MatType& data,
                             const size_t begin,
                             const size_t count,
                             const size_t splitDimension,
                             const double splitVal,
                             std::vector<size_t>& oldFromNew);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "mean_split_impl.hpp"

#endif
