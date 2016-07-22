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
 * The vantage point tree is a kind of the binary space tree. In contrast to
 * BinarySpaceTree, each left intermediate node of the vantage point tree
 * contains a point which is the centroid of the node. When recursively
 * splitting nodes, the VPTree class selects a vantage point and splits the node
 * according to the distance to this point. Thus, points that are closer to the
 * vantage point form the left subtree (and the vantage point is the only point
 * that the left node contains). Other points form the right subtree.
 * In such a way, the bound of each left node is a ball and the vantage point is
 * the centroid of the bound. The bound of each right node is a hollow ball
 * centered at the vantage point.
 * 
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{yianilos1993vptrees,
 *   author = {Yianilos, Peter N.},
 *   title = {Data Structures and Algorithms for Nearest Neighbor Search in
 *      General Metric Spaces},
 *   booktitle = {Proceedings of the Fourth Annual ACM-SIAM Symposium on
 *      Discrete Algorithms},
 *   series = {SODA '93},
 *   year = {1993},
 *   isbn = {0-89871-313-7},
 *   pages = {311--321},
 *   numpages = {11},
 *   publisher = {Society for Industrial and Applied Mathematics},
 *   address = {Philadelphia, PA, USA}
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
