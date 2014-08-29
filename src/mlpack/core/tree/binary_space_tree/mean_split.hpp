/**
 * @file mean_split.hpp
 * @author Yash Vadalia
 * @author Ryan Curtin
 *
 * Definition of MeanSplit, a class that splits a binary space partitioning tree
 * node into two parts using the mean of the values in a certain dimension.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_MEAN_SPLIT_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_MEAN_SPLIT_HPP

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
                        size_t& splitDimension,
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
                        size_t& splitDimension,
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

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "mean_split_impl.hpp"

#endif
