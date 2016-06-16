/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Typedefs of RectangleTrees, for use by classes that require trees matching
 * the TreeType API.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_TYPEDEF_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_TYPEDEF_HPP

#include "rectangle_tree.hpp"

namespace mlpack {
namespace tree {

/**
 * An implementation of the R tree that satisfies the TreeType policy API.
 *
 * This is the same R-tree structure as proposed by Guttman:
 *
 * @code
 * @inproceedings{guttman1984r,
 *   title={R-trees: a dynamic index structure for spatial searching},
 *   author={Guttman, A.},
 *   booktitle={Proceedings of the 1984 ACM SIGMOD International Conference on
 *       Management of Data (SIGMOD '84)},
 *   volume={14},
 *   number={2},
 *   year={1984},
 *   publisher={ACM}
 * }
 * @endcode
 *
 * @see @ref trees, RStarTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using RTree = RectangleTree<MetricType,
                            StatisticType,
                            MatType,
                            RTreeSplit,
                            RTreeDescentHeuristic>;

/**
 * The R*-tree, a more recent variant of the R tree.  This template typedef
 * satisfies the TreeType policy API.
 *
 * @code
 * @inproceedings{beckmann1990r,
 *   title={The R*-tree: an efficient and robust access method for points and
 *       rectangles},
 *   author={Beckmann, N. and Kriegel, H.-P. and Schneider, R. and Seeger, B.},
 *   booktitle={Proceedings of the 1990 ACM SIGMOD International Conference on
 *       Management of Data (SIGMOD '90)},
 *   volume={19},
 *   number={2},
 *   year={1990},
 *   publisher={ACM}
 * }
 * @endcode
 *
 * @see @ref trees, RTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using RStarTree = RectangleTree<MetricType,
                                StatisticType,
                                MatType,
                                RStarTreeSplit,
                                RStarTreeDescentHeuristic>;

/**
 * The X-tree, a variant of the R tree with supernodes.  This template typedef
 * satisfies the TreeType policy API.
 *
 * @code
 * @inproceedings{berchtold1996r,
 *   title = {The X-Tree: An Index Structure for High--Dimensional Data},
 *   author = {Berchtold, Stefan and Keim, Daniel A. and Kriegel, Hans-Peter},
 *   booktitle = {Proc. 22th Int. Conf. on Very Large Databases (VLDB'96), Bombay, India},
 *   editor = {Vijayaraman, T. and Buchmann, Alex and Mohan, C. and Sarda, N.},
 *   pages = {28--39},
 *   year = {1996},
 *   publisher = {Morgan Kaufmann}
 * }
 * @endcode
 *
 * @see @ref trees, RTree, RStarTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using XTree = RectangleTree<MetricType,
                            StatisticType,
                            MatType,
                            XTreeSplit,
                            RTreeDescentHeuristic>;

} // namespace tree
} // namespace mlpack

#endif
