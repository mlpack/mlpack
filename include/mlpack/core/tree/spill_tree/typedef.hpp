/**
 * @file core/tree/spill_tree/typedef.hpp
 * @author Ryan Curtin
 * @author Marcos Pividori
 *
 * Template typedefs for the SpillTree class that satisfy the requirements
 * of the TreeType policy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_TYPEDEF_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_TYPEDEF_HPP

#include "../space_split/mean_space_split.hpp"
#include "../space_split/midpoint_space_split.hpp"

namespace mlpack {

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
template<typename DistanceType, typename StatisticType, typename MatType>
using SPTree = SpillTree<DistanceType,
                         StatisticType,
                         MatType,
                         AxisOrthogonalHyperplane,
                         MidpointSpaceSplit>;

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
template<typename DistanceType, typename StatisticType, typename MatType>
using MeanSPTree = SpillTree<DistanceType,
                             StatisticType,
                             MatType,
                             AxisOrthogonalHyperplane,
                             MeanSpaceSplit>;

/**
 * A hybrid spill tree considering general splitting hyperplanes (not
 * necessarily axis-orthogonal).  This particular implementation will consider
 * the midpoint of the projection of the data in the vector determined by the
 * farthest pair of points.  This can sometimes give better performance, but
 * generally it doesn't because it takes O(d) to calculate the projection of the
 * query point when deciding which node to traverse, while when using a
 * axis-orthogonal hyperplane, as SPTree does, we can do it in O(1).
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, SpillTree, SPTree
 */
template<typename DistanceType, typename StatisticType, typename MatType>
using NonOrtSPTree = SpillTree<DistanceType,
                               StatisticType,
                               MatType,
                               Hyperplane,
                               MidpointSpaceSplit>;

/**
 * A mean-split hybrid spill tree considering general splitting hyperplanes (not
 * necessarily axis-orthogonal).  This is the same as the NonOrtSPTree, but this
 * particular implementation will use the mean of the data in the split
 * projection as the value on which to split, instead of the midpoint.
 * This can sometimes give better performance, but it is not always clear which
 * type of tree is best.
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, SpillTree, MeanSPTree, NonOrtSPTree
 */
template<typename DistanceType, typename StatisticType, typename MatType>
using NonOrtMeanSPTree = SpillTree<DistanceType,
                                   StatisticType,
                                   MatType,
                                   Hyperplane,
                                   MeanSpaceSplit>;

} // namespace mlpack

#endif
