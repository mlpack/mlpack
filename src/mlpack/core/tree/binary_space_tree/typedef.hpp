/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Template typedefs for the BinarySpaceTree class that satisfy the requirements
 * of the TreeType policy class.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP

// In case it hasn't been included yet.
#include "../binary_space_tree.hpp"

namespace mlpack {
namespace tree {

/**
 * midpoint split kd-tree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using KDTree = BinarySpaceTree<MetricType,
                               StatisticType,
                               MatType,
                               bound::HRectBound,
                               MidpointSplit>;

/**
 * mean split kd-tree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using MeanSplitKDTree = BinarySpaceTree<MetricType,
                                        StatisticType,
                                        MatType,
                                        bound::HRectBound,
                                        MeanSplit>;

/**
 * midpoint split ball tree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using BallTree = BinarySpaceTree<MetricType,
                                 StatisticType,
                                 MatType,
                                 bound::HRectBound,
                                 MidpointSplit>;

/**
 * mean split ball tree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using MeanSplitBallTree = BinarySpaceTree<MetricType,
                                          StatisticType,
                                          MatType,
                                          bound::HRectBound,
                                          MeanSplit>;

} // namespace tree
} // namespace mlpack

#endif
