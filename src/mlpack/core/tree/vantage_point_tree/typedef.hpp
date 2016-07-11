/**
 * @file typedef.hpp
 *
 * Template typedefs for the VantagePointTree class that satisfy the
 * requirements of the TreeType policy class.
 */
#ifndef MLPACK_CORE_TREE_VANTAGE_POINT_TREE_TYPEDEF_HPP
#define MLPACK_CORE_TREE_VANTAGE_POINT_TREE_TYPEDEF_HPP

// In case it hasn't been included yet.
#include "../vantage_point_tree.hpp"

namespace mlpack {
namespace tree {
  
template<typename MetricType, typename StatisticType, typename MatType>
using VPTree = VantagePointTree<MetricType,
                                          StatisticType,
                                          MatType,
                                          bound::HollowBallBound,
                                          VantagePointSplit>;

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_TYPEDEF_HPP
