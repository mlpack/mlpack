/**
 * @file vantage_point_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of class VantagePointSplit, a class that splits a vantage point
 * tree into two parts using the distance to a certain vantage point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/perform_split.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * The class splits a binary space partitioning tree node according to the
 * median distance to the vantage point. Thus points that are closer to the
 * vantage point belong to the left subtree and points that are farther from
 * the vantage point belong to the right subtree.
 */
template<typename BoundType,
         typename MatType = arma::mat,
         size_t MaxNumSamples = 100>
class VantagePointSplit
{
 public:
  //! The matrix element type.
  typedef typename MatType::elem_type ElemType;
  //! The bounding shape type.
  typedef typename BoundType::MetricType MetricType;
  //! A struct that contains an information about the split.
  struct SplitInfo
  {
    //! The vantage point.
    arma::Col<ElemType> vantagePoint;
    //! The median distance according to which the node will be split.
    ElemType mu;
    //! An instance of the MetricType class.
    const MetricType* metric;

    SplitInfo() :
        mu(0),
        metric(NULL)
    { }

    template<typename VecType>
    SplitInfo(const MetricType& metric, const VecType& vantagePoint,
        ElemType mu) :
        vantagePoint(vantagePoint),
        mu(mu),
        metric(&metric)
    { }
  };

  /**
   * Split the node according to the distance to a vantage point.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitInfo An information about the split. This information contains
   *    the vantage point and the median distance to the vantage point.
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
    return split::PerformSplit<MatType, VantagePointSplit>(data, begin, count,
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
    return split::PerformSplit<MatType, VantagePointSplit>(data, begin, count,
        splitInfo, oldFromNew);
  }

  /**
   * Indicates that a point should be assigned to the left subtree.
   * This method returns true if a point should be assigned to the left subtree,
   * i.e., if the distance from the point to the vantage point is less then the
   * median value. Otherwise it returns false.
   *
   * @param point The point that is being assigned.
   * @param splitInfo An information about the split.
   */
  template<typename VecType>
  static bool AssignToLeftNode(const VecType& point,
                               const SplitInfo& splitInfo)
  {
    return (splitInfo.metric->Evaluate(splitInfo.vantagePoint, point) <
        splitInfo.mu);
  }

 private:
  /**
   * Select the best vantage point, i.e., the point with the largest second
   * moment of the distance from a number of random node points to the vantage
   * point.  Firstly this method selects no more than MaxNumSamples random
   * points.  Then it evaluates each point, i.e., calculates the corresponding
   * second moment and selects the point with the largest moment. Each random
   * point belongs to the node.
   *
   * @param metric The metric used by the tree.
   * @param data The dataset used by the tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param vantagePoint The index of the vantage point in the dataset.
   * @param mu The median value of distance form the vantage point to
   * a number of random points.
   */
  static void SelectVantagePoint(const MetricType& metric,
                                 const MatType& data,
                                 const size_t begin,
                                 const size_t count,
                                 size_t& vantagePoint,
                                 ElemType& mu);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "vantage_point_split_impl.hpp"

#endif  //  MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
