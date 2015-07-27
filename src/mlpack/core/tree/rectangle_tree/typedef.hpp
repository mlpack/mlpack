/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Typedefs of RectangleTrees, for use by classes that require trees matching
 * the TreeType API.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_TYPEDEF_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_TYPEDEF_HPP

#include "rectangle_tree.hpp"

namespace mlpack {
namespace tree {

/**
 * regular r-tree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using RTree = RectangleTree<MetricType,
                            StatisticType,
                            MatType,
                            RTreeSplit,
                            RTreeDescentHeuristic>;

/**
 * r*-tree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using RStarTree = RectangleTree<MetricType,
                                StatisticType,
                                MatType,
                                RStarTreeSplit,
                                RStarTreeDescentHeuristic>;

/**
 * X-tree
 * (not yet finished)
 */
//template<typename MetricType, typename StatisticType, typename MatType>
//using XTree = RectangleTree<MetricType,
//                            StatisticType,
//                            MatType,
//                            XTreeSplit,
//                            XTreeDescentHeuristic>;

} // namespace tree
} // namespace mlpack

#endif
