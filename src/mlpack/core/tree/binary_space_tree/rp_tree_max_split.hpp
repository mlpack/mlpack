/**
 * @file rp_tree_max_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of class (RPTreeMaxSplit) to split a binary space partition
 * tree.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * This class splits a node by a random hyperplane. In order to choose the
 * hyperplane we need to choose the normal to the hyperplane and the position
 * of the hyperplane i.e. the scalar product of the normal and a point.
 *
 * A point will be assigned to the left subtree if the product of
 * this point and the normal is less or equal to the split value (i.e. the
 * position of the hyperplane).
 */
template<typename BoundType, typename MatType = arma::mat>
class RPTreeMaxSplit
{
 public:
  typedef typename MatType::elem_type ElemType;
  struct SplitInfo
  {
    //! The normal vector to the hyperplane that splits the node.
    arma::Col<ElemType> direction;
    //! The value according to which the node is being split.
    ElemType splitVal;
  };
  /**
   * Split the node by a random hyperplane.
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
   * Get random deviation from the median of points multiplied by the direction
   * obtained in GetRandomDirection().
   *
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param direction A random unit vector.
   */
  static ElemType GetRandomDeviation(const MatType& data,
                                     const size_t begin,
                                     const size_t count,
                                     const arma::Col<ElemType>& direction);

  /**
   * Obtain a number of random distinct samples from the dataset. All samples
   * belong to [begin, begin + count).
   *
   * @param distinctSamples The indices of the samples.
   * @param begin The lower bound of indices.
   * @param count The number of point candidates.
   * @param numSamples The maximum number of samples.
   */
  static void GetDistinctSamples(arma::uvec& distinctSamples,
                                 const size_t begin,
                                 const size_t count,
                                 const size_t numSamples);

  /**
   * This method finds the position of the hyperplane that will split the node.
   *
   * @param data The dataset used by the binary space tree.
   * @param begin Index of the starting point in the dataset that belongs to
   *    this node.
   * @param count Number of points in this node.
   * @param direction A random vector that is the normal to the hyperplane
   *    which will split the node.
   * @param splitVal The value according which the node will be split.
   */
  static bool GetSplitVal(const MatType& data,
                          const size_t begin,
                          const size_t count,
                          const arma::Col<ElemType>& direction,
                          ElemType& splitVal);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "rp_tree_max_split_impl.hpp"

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_HPP
