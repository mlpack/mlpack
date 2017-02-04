/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Template typedefs for the BinarySpaceTree class that satisfy the requirements
 * of the TreeType policy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_TYPEDEF_HPP

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
                                 bound::BallBound,
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
                                          bound::BallBound,
                                          MeanSplit>;

/**
 * The vantage point tree (which is also called the metric tree. Vantage point
 * trees and metric trees were invented independently by Yianilos an Uhlmann) is
 * a kind of the binary space tree. When recursively splitting nodes, the VPTree
 * class selects the vantage point and splits the node according to the distance
 * to this point. Thus, points that are closer to the vantage point form the
 * inner subtree. Other points form the outer subtree. The vantage point is
 * contained in the first (inner) node.
 *
 * This implementation differs from the original algorithms. Namely, vantage
 * points are not contained in intermediate nodes. The tree has points only in
 * the leaves of the tree.
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
template<typename BoundType,
         typename MatType = arma::mat>
using VPTreeSplit = VantagePointSplit<BoundType, MatType, 100>;

template<typename MetricType, typename StatisticType, typename MatType>
using VPTree = BinarySpaceTree<MetricType,
                               StatisticType,
                               MatType,
                               bound::HollowBallBound,
                               VPTreeSplit>;

/**
 * A max-split random projection tree. When recursively splitting nodes, the
 * MaxSplitRPTree class selects a random hyperplane and splits a node by the
 * hyperplane. The tree holds points in leaf nodes. In contrast to the k-d tree,
 * children of a MaxSplitRPTree node may overlap.
 *
 * @code
 * @inproceedings{dasgupta2008,
 *   author = {Dasgupta, Sanjoy and Freund, Yoav},
 *   title = {Random Projection Trees and Low Dimensional Manifolds},
 *   booktitle = {Proceedings of the Fortieth Annual ACM Symposium on Theory of
 *       Computing},
 *   series = {STOC '08},
 *   year = {2008},
 *   pages = {537--546},
 *   numpages = {10},
 *   publisher = {ACM},
 *   address = {New York, NY, USA},
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, BallTree, MeanSplitKDTree
 */

template<typename MetricType, typename StatisticType, typename MatType>
using MaxRPTree = BinarySpaceTree<MetricType,
                                  StatisticType,
                                  MatType,
                                  bound::HRectBound,
                                  RPTreeMaxSplit>;

/**
 * A mean-split random projection tree. When recursively splitting nodes, the
 * RPTree class may perform one of two different kinds of split.
 * Depending on the diameter and the average distance between points, the node
 * may be split by a random hyperplane or according to the distance from the
 * mean point. The tree holds points in leaf nodes. In contrast to the k-d tree,
 * children of a MaxSplitRPTree node may overlap.
 *
 * @code
 * @inproceedings{dasgupta2008,
 *   author = {Dasgupta, Sanjoy and Freund, Yoav},
 *   title = {Random Projection Trees and Low Dimensional Manifolds},
 *   booktitle = {Proceedings of the Fortieth Annual ACM Symposium on Theory of
 *       Computing},
 *   series = {STOC '08},
 *   year = {2008},
 *   pages = {537--546},
 *   numpages = {10},
 *   publisher = {ACM},
 *   address = {New York, NY, USA},
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, BallTree, MeanSplitKDTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using RPTree = BinarySpaceTree<MetricType,
                                  StatisticType,
                                  MatType,
                                  bound::HRectBound,
                                  RPTreeMeanSplit>;

/**
 * The Universal B-tree. When recursively splitting nodes, the class
 * calculates addresses of all points and splits each node according to the
 * median address. Children may overlap since the implementation
 * of a tighter bound requires a lot of arithmetic operations. In order to get
 * a tighter bound increase the CellBound::maxNumBounds constant.
 *
 * @code
 * @inproceedings{bayer1997,
 *   author = {Bayer, Rudolf},
 *   title = {The Universal B-Tree for Multidimensional Indexing: General
 *       Concepts},
 *   booktitle = {Proceedings of the International Conference on Worldwide
 *       Computing and Its Applications},
 *   series = {WWCA '97},
 *   year = {1997},
 *   isbn = {3-540-63343-X},
 *   pages = {198--209},
 *   numpages = {12},
 *   publisher = {Springer-Verlag},
 *   address = {London, UK, UK},
 * }
 * @endcode
 *
 * This template typedef satisfies the TreeType policy API.
 *
 * @see @ref trees, BinarySpaceTree, BallTree, MeanSplitKDTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using UBTree = BinarySpaceTree<MetricType,
                               StatisticType,
                               MatType,
                               bound::CellBound,
                               UBTreeSplit>;

} // namespace tree
} // namespace mlpack

#endif
