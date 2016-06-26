/**
 * @file vantage_point_split.hpp
 * @author Mikhail Lozhnikov
 *
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename BoundType, typename MatType = arma::mat>
class VantagePointSplit
{
 public:
  typedef typename MatType::elem_type ElemType;
  /**
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
  static const size_t maxNumSamples = 1000;

  template<typename StructElemType>
  struct SortStruct
  {
    size_t point;
    size_t n;
    ElemType dist;
  };

  template<typename StructElemType>
  static bool StructComp(const SortStruct<StructElemType>& s1,
      const SortStruct<StructElemType>& s2)
  {
    return (s1.dist < s2.dist);
  };

  static void SelectVantagePoint(const BoundType& bound, const MatType& data,
    const size_t begin, const size_t count, size_t& vantagePoint, ElemType& mu);

  static void GetDistinctSamples(arma::uvec& distinctSamples,
      const size_t numSamples, const size_t begin, const size_t upperBound);

  static void GetMedian(const BoundType& bound, const MatType& data,
      const arma::uvec& samples,  const size_t vantagePoint, ElemType& mu);

  static ElemType GetSecondMoment(const BoundType& bound, const MatType& data,
      const arma::uvec& samples,  const size_t vantagePoint);

  static bool IsContainedInBall(const BoundType& bound, const MatType& mat,
      const size_t vantagePoint, const size_t point, const ElemType mu);

  static size_t PerformSplit(const BoundType& bound,
                             MatType& data,
                             const size_t begin,
                             const size_t count,
                             const size_t vantagePoint,
                             const ElemType mu);

  static size_t PerformSplit(const BoundType& bound,
                             MatType& data,
                             const size_t begin,
                             const size_t count,
                             const size_t vantagePoint,
                             const ElemType mu,
                             std::vector<size_t>& oldFromNew);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "vantage_point_split_impl.hpp"

#endif  //  MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_HPP
