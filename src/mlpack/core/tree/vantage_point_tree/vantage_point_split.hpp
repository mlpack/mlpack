/**
 * @file vantage_point_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of class VantagePointSplit, a class that splits a vantage point
 * tree into two parts using the distance to a certain vantage point.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename BoundType,
         typename MatType = arma::mat,
         size_t maxNumSamples = 100>
class VantagePointSplit
{
 public:
  typedef typename MatType::elem_type ElemType;
  typedef typename BoundType::MetricType MetricType;
  /**
   * Split the node according to the distance to a vantage point.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param splitCol The index at which the dataset is divided into two parts
   *    after the rearrangement.
   */
  static bool SplitNode(const BoundType& bound,
                        MatType& data,
                        const size_t begin,
                        const size_t count,
                        size_t& splitCol);

  /**
   * Split the node according to the distance to a vantage point.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
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
   * Select the best vantage point i.e. the point with the largest second moment
   * of the distance from a number of random node points to the vantage point.
   * Firstly this methods selects no more than maxNumSamples random points.
   * Then it evaluates each point i.e. calcilates the corresponding second
   * moment and selects the point with the largest moment. Each random point
   * belongs to the node.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param vantagePoint The index of the vantage point in the dataset.
   * @param mu The median value of distance form the vantage point to
   * a number of random points.
   */
  static void SelectVantagePoint(const MetricType& metric, const MatType& data,
    const size_t begin, const size_t count, size_t& vantagePoint, ElemType& mu);

  /**
   * This method returns true if a point should be assigned to the left subtree
   * i.e. the distance from the point to the vantage point is less then
   * the median value. Otherwise it returns false.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the binary space tree.
   * @param vantagePoint The vantage point.
   * @param point The point that is being assigned.
   * @param mu The median value.
   */
  template<typename VecType>
  static bool AssignToLeftSubtree(const MetricType& metric, const MatType& mat,
      const VecType& vantagePoint, const size_t point, const ElemType mu)
  {
    return (metric.Evaluate(vantagePoint, mat.col(point)) < mu);
  }

  /**
   * Perform split according to the median value and the vantage point.
   * 
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param vantagePoint The vantage point.
   * @param mu The median value.
   */
  template<typename VecType>
  static size_t PerformSplit(const MetricType& metric,
                             MatType& data,
                             const size_t begin,
                             const size_t count,
                             const VecType& vantagePoint,
                             const ElemType mu);

  /**
   * Perform split according to the median value and the vantage point.
   * 
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param vantagePoint The vantage point.
   * @param mu The median value.
   * @param oldFromNew Vector which will be filled with the old positions for
   *    each new point.
   */
  template<typename VecType>
  static size_t PerformSplit(const MetricType& metric,
                             MatType& data,
                             const size_t begin,
                             const size_t count,
                             const VecType& vantagePoint,
                             const ElemType mu,
                             std::vector<size_t>& oldFromNew);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "vantage_point_split_impl.hpp"

#endif  //  MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
