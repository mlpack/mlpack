/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Typedefs of RectangleTrees, for use by classes that require trees matching
 * the TreeType API.
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
                            RTreeDescentHeuristic,
                            NoAuxiliaryInformation>;

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
                                RStarTreeDescentHeuristic,
                                NoAuxiliaryInformation>;

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
                            RTreeDescentHeuristic,
                            XTreeAuxiliaryInformation>;

/**
 * The Hilbert R-tree, a variant of the R tree with an ordering along
 * the Hilbert curve. This template typedef satisfies the TreeType policy API.
 *
 * @code
 * @inproceedings{kamel1994r,
 *   author = {Kamel, Ibrahim and Faloutsos, Christos},
 *   title = {Hilbert R-tree: An Improved R-tree Using Fractals},
 *   booktitle = {Proceedings of the 20th International Conference on Very Large Data Bases},
 *   series = {VLDB '94},
 *   year = {1994},
 *   isbn = {1-55860-153-8},
 *   pages = {500--509},
 *   numpages = {10},
 *   url = {http://dl.acm.org/citation.cfm?id=645920.673001},
 *   acmid = {673001},
 *   publisher = {Morgan Kaufmann Publishers Inc.},
 *   address = {San Francisco, CA, USA}
 * }
 * @endcode
 *
 * @see @ref trees, RTree, DiscreteHilbertRTree
 */
template<typename TreeType>
using DiscreteHilbertRTreeAuxiliaryInformation =
      HilbertRTreeAuxiliaryInformation<TreeType,DiscreteHilbertValue>;

template<typename MetricType, typename StatisticType, typename MatType>
using HilbertRTree = RectangleTree<MetricType,
                            StatisticType,
                            MatType,
                            HilbertRTreeSplit<2>,
                            HilbertRTreeDescentHeuristic,
                            DiscreteHilbertRTreeAuxiliaryInformation>;

template<typename MetricType, typename StatisticType, typename MatType>
using RPlusTree = RectangleTree<MetricType,
                            StatisticType,
                            MatType,
                            RPlusTreeSplit<RPlusTreeSplitPolicy,
                                           MinimalCoverageSweep>,
                            RPlusTreeDescentHeuristic,
                            NoAuxiliaryInformation>;

template<typename MetricType, typename StatisticType, typename MatType>
using RPlusPlusTree = RectangleTree<MetricType,
                            StatisticType,
                            MatType,
                            RPlusTreeSplit<RPlusPlusTreeSplitPolicy,
                                           MinimalCoverageSweep>,
                            RPlusPlusTreeDescentHeuristic,
                            RPlusPlusTreeAuxiliaryInformation>;
} // namespace tree
} // namespace mlpack

#endif
