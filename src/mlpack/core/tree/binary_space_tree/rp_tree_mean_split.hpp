/**
 * @file rp_tree_mean_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of class (RPTreeMaxSplit) to split a binary space partition
 * tree.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP

#include <mlpack/core.hpp>
#include "rp_tree_max_split.hpp"

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
  typedef typename MatType::elem_type ElemType;
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
  static bool SplitNode(const BoundType& /*bound*/,
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
  static bool AssignToLeftNode(
    const VecType& point,
    const SplitInfo& splitInfo)
  {
    if (splitInfo.meanSplit)
      return arma::dot(point - splitInfo.mean, point - splitInfo.mean) <=
          splitInfo.splitVal;

    return (arma::dot(point, splitInfo.direction) <= splitInfo.splitVal);
  }

 private:

  /**
   * Get a random unit vector of size direction.n_elem.
   *
   * @param direction The variable into which the method saves the vector.
   */
  static void GetRandomDirection(arma::Col<ElemType>& direction);

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

  friend RPTreeMaxSplit<BoundType, MatType>;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "rp_tree_mean_split_impl.hpp"

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_HPP
