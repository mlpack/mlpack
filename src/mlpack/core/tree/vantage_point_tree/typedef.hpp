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

/**
 * The vantage point tree (which is also called the metric tree. Vantage point
 * trees and metric trees were invented independently by Yianilos an Uhlmann) is
 * a kind of the binary space tree. In contrast to BinarySpaceTree, each
 * intermediate node of the vantage point tree contains three children.
 * The first child contains exactly one point (the vantage point). When
 * recursively splitting nodes, the VPTree class selects the vantage point and
 * splits the node according to the distance to this point. Thus, points that
 * are closer to the vantage point form the inner subtree. Other points form the
 * outer subtree. The vantage point is contained in the first (central) node.
 * In such a way, the bound of each inner and outer nodes is a hollow ball and
 * the vantage point is the centroid of the bound.
 *
 * This implementation differs from the original algorithms. Namely, the central
 * node was introduced in order to simplify dual-tree traversers.
 * 
 * For more information, see the following papers.
 *
 * @code
 * @inproceedings{yianilos1993vptrees,
 *   author = {Yianilos, Peter N.},
 *   title = {Data Structures and Algorithms for Nearest Neighbor Search in
 *       General Metric Spaces},
 *   booktitle = {Proceedings of the Fourth Annual ACM-SIAM Symposium on
 *       Discrete Algorithms},
 *   series = {SODA '93},
 *   year = {1993},
 *   isbn = {0-89871-313-7},
 *   pages = {311--321},
 *   numpages = {11},
 *   publisher = {Society for Industrial and Applied Mathematics},
 *   address = {Philadelphia, PA, USA}
 * }
 *
 * @article{uhlmann1991metrictrees,
 *   author = {Jeffrey K. Uhlmann},
 *   title = {Satisfying general proximity / similarity queries with metric
 *       trees},
 *   journal = {Information Processing Letters},
 *   volume = {40},
 *   number = {4},
 *   pages = {175 - 179},
 *   year = {1991},
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, VantagePointTree, VPTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using VPTree = VantagePointTree<MetricType,
                                StatisticType,
                                MatType,
                                bound::HollowBallBound,
                                VantagePointSplit>;

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_TYPEDEF_HPP
