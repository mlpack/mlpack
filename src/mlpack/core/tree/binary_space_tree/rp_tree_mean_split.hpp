/**
 * @file rp_tree_mean_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of class (RPTreeMaxSplit) to split a binary space partition
 * tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP

#include <mlpack/core.hpp>
#include "rp_tree_max_split.hpp"
#include <mlpack/core/tree/perform_split.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * This class splits a binary space tree. This class provides two different
 * kinds of split. The mean split (i.e. all points are split by the median
 * of their distance to the mean point) is performed if the average distance
 * between points multiplied by a constant is greater than the diameter of the
 * node. Otherwise, the median split (i.e. the node is split by a random
 * hyperplane) is performed.
 */
template<typename BoundType, typename MatType = arma::mat>
class RPTreeMeanSplit
{
 public:
  //! The element type held by the matrix type.
  typedef typename MatType::elem_type ElemType;
  //! An information about the partition.
  struct SplitInfo
  {
    //! The normal to the hyperplane that will split the node.
    arma::Col<ElemType> direction;
    //! The mean of some sampled points.
    arma::Col<ElemType> mean;
    //! The value according to which the split will be performed.
    ElemType splitVal;
    //! Indicates that we should use the mean split algorithm instead of the
    //! median split.
    bool meanSplit;
  };

  /**
   * Split the node according to the mean value in the dimension with maximum
   * width.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitInfo An information about the split. This information contains
   *    the direction and the value.
   */
  static bool SplitNode(const BoundType& /* bound */,
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
    return split::PerformSplit<MatType, RPTreeMeanSplit>(data, begin, count,
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
    return split::PerformSplit<MatType, RPTreeMeanSplit>(data, begin, count,
        splitInfo, oldFromNew);
  }

  /**
   * Indicates that a point should be assigned to the left subtree.
   *
   * @param point The point that is being assigned.
   * @param splitInfo An information about the split.
   */
  template<typename VecType>
  static bool AssignToLeftNode(const VecType& point, const SplitInfo& splitInfo)
  {
    if (splitInfo.meanSplit)
      return arma::dot(point - splitInfo.mean, point - splitInfo.mean) <=
          splitInfo.splitVal;

    return (arma::dot(point, splitInfo.direction) <= splitInfo.splitVal);
  }

 private:

  /**
   * Get the average distance between points in the dataset.
   *
   * @param data The dataset used by the binary space tree.
   * @param samples The indices of points that will be used for the calculation.
   */
  static ElemType GetAveragePointDistance(MatType& data,
                                          const arma::uvec& samples);

  /**
   * Get the median of scalar products of the sampled points and the normal
   * to the hyperplane (i.e. the position of the hyperplane).
   *
   * @param data The dataset used by the binary space tree.
   * @param samples The indices of points that will be used for the calculation.
   * @param direction The normal to the hyperplane.
   * @param splitVal The median value.
   */
  static bool GetDotMedian(const MatType& data,
                           const arma::uvec& samples,
                           const arma::Col<ElemType>& direction,
                           ElemType& splitVal);

  /**
   * Get the mean point and the median of distance from the mean to any point of
   * the dataset.
   *
   * @param data The dataset used by the binary space tree.
   * @param samples The indices of points that will be used for the calculation.
   * @param mean The mean point.
   * @param splitVal The median value.
   */
  static bool GetMeanMedian(const MatType& data,
                            const arma::uvec& samples,
                            arma::Col<ElemType>& mean,
                            ElemType& splitVal);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "rp_tree_mean_split_impl.hpp"

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP
