/**
 * @file typedef.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Template typedefs for the SpillTree class that satisfy the requirements
 * of the TreeType policy class.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_TYPEDEF_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_TYPEDEF_HPP

// In case it hasn't been included yet.
#include "../spill_tree.hpp"
#include "../binary_space_tree/midpoint_split.hpp"
#include "../binary_space_tree/mean_split.hpp"

namespace mlpack {
namespace tree {

/**
 * The hybrid spill tree.  It is a variant of metric-trees in which the children
 * of a node can "spill over" onto each other, and contain shared datapoints.
 *
 * When recursively splitting nodes, the SPTree class select the dimension with
 * maximum width to split on, and picks the midpoint of the range in that
 * dimension as the value on which to split nodes.
 *
 * In each case a "overlapping buffer" is defined, included points at a distance
 * less than tau from the decision boundary defined by the midpoint.
 *
 * For each node, we first split the points considering the overlapping buffer.
 * If either of its children contains more than rho fraction of the total points
 * we undo the overlapping splitting. Instead a conventional partition is used.
 * In this way, we can ensure that each split reduces the number of points of a
 * node by at least a constant factor.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{
 *   author = {Ting Liu, Andrew W. Moore, Alexander Gray and Ke Yang},
 *   title = {An Investigation of Practical Approximate Nearest Neighbor
 *     Algorithms},
 *   booktitle = {Advances in Neural Information Processing Systems 17},
 *   year = {2005},
 *   pages = {825--832}
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, SpillTree, MeanSPTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using SPTree = SpillTree<MetricType,
                         StatisticType,
                         MatType,
                         MidpointSplit>;

/**
 * A mean-split hybrid spill tree.  This is the same as the SPTree, but this
 * particular implementation will use the mean of the data in the split
 * dimension as the value on which to split, instead of the midpoint.
 * This can sometimes give better performance, but it is not always clear which
 * type of tree is best.
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, SpillTree, SPTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using MeanSPTree = SpillTree<MetricType,
                             StatisticType,
                             MatType,
                             MeanSplit>;

} // namespace tree
} // namespace mlpack

#endif
