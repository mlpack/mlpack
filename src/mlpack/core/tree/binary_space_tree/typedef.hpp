/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Template typedefs for the BinarySpaceTree class that satisfy the requirements
 * of the TreeType policy class.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP

// In case it hasn't been included yet.
#include "../binary_space_tree.hpp"

namespace mlpack {
namespace tree {

/**
 * The standard midpoint-split kd-tree.  This is not the original formulation by
 * Bentley but instead the later formulation by Deng and Moore, which only holds
 * points in the leaves of the tree.  When recursively splitting nodes, the
 * KDTree class select the dimension with maximum variance to split on, and
 * picks the midpoint of the range in that dimension as the value on which to
 * split nodes.
 *
 * For more information, see the following papers.
 *
 * @code
 * @article{bentley1975multidimensional,
 *   title={Multidimensional binary search trees used for associative searching},
 *   author={Bentley, J.L.},
 *   journal={Communications of the ACM},
 *   volume={18},
 *   number={9},
 *   pages={509--517},
 *   year={1975},
 *   publisher={ACM}
 * }
 *
 * @inproceedings{deng1995multiresolution,
 *   title={Multiresolution instance-based learning},
 *   author={Deng, K. and Moore, A.W.},
 *   booktitle={Proceedings of the 1995 International Joint Conference on AI
 *       (IJCAI-95)},
 *   pages={1233--1239},
 *   year={1995}
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, MeanSplitKDTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using KDTree = BinarySpaceTree<MetricType,
                               StatisticType,
                               MatType,
                               bound::HRectBound,
                               MidpointSplit>;

/**
 * A mean-split kd-tree.  This is the same as the KDTree, but this particular
 * implementation will use the mean of the data in the split dimension as the
 * value on which to split, instead of the midpoint.  This can sometimes give
 * better performance, but it is not always clear which type of tree is best.
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, KDTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using MeanSplitKDTree = BinarySpaceTree<MetricType,
                                        StatisticType,
                                        MatType,
                                        bound::HRectBound,
                                        MeanSplit>;

/**
 * A midpoint-split ball tree.  This tree holds its points only in the leaves,
 * similar to the KDTree and MeanSplitKDTree.  However, the bounding shape of
 * each node is a ball, not a hyper-rectangle.  This can make the ball tree
 * advantageous in some higher-dimensional situations and for some datasets.
 * The tree construction algorithm here is the same as Omohundro's 'K-d
 * construction algorithm', except the splitting value is the midpoint, not the
 * median.  This can result in trees that better reflect the data, although they
 * may be unbalanced.
 *
 * @code
 * @techreport{omohundro1989five,
 *   author={S.M. Omohundro},
 *   title={Five balltree construction algorithms},
 *   year={1989},
 *   institution={University of California, Berkeley International Computer
 *       Science Institute Technical Reports},
 *   number={TR-89-063}
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, KDTree, MeanSplitBallTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using BallTree = BinarySpaceTree<MetricType,
                                 StatisticType,
                                 MatType,
                                 bound::HRectBound,
                                 MidpointSplit>;

/**
 * A mean-split ball tree.  This tree, like the BallTree, holds its points only
 * in the leaves.  The tree construction algorithm here is the same as
 * Omohundro's 'K-dc onstruction algorithm', except the splitting value is the
 * mean, not the median.  This can result in trees that better reflect the data,
 * although they may be unbalanced.
 *
 * @code
 * @techreport{omohundro1989five,
 *   author={S.M. Omohundro},
 *   title={Five balltree construction algorithms},
 *   year={1989},
 *   institution={University of California, Berkeley International Computer
 *       Science Institute Technical Reports},
 *   number={TR-89-063}
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, BallTree, MeanSplitKDTree
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
