/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Typedef of cover tree to match TreeType API.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP

#include "cover_tree.hpp"

namespace mlpack {
namespace tree {

/**
 * The standard cover tree, as detailed in the original cover tree paper:
 *
 * @code
 * @inproceedings{
 *   author={Beygelzimer, A. and Kakade, S. and Langford, J.},
 *   title={Cover trees for nearest neighbor},
 *   booktitle={Proceedings of the 23rd International Conference on Machine
 *       Learning (ICML 2006)},
 *   pages={97--104},
 *   year={2006}
 * }
 * @endcode
 *
 * This template typedef satisfies the requirements of the TreeType API.
 *
 * @see @ref trees, CoverTree
 */
template<typename MetricType, typename StatisticType, typename MatType>
using StandardCoverTree = CoverTree<MetricType,
                                    StatisticType,
                                    MatType,
                                    FirstPointIsRoot>;

} // namespace tree
} // namespace mlpack

#endif
