/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Typedef of cover tree to match TreeType API.
 *
 * This file is part of mlpack 2.0.2.
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
#ifndef MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP
#define MLPACK_CORE_TREE_COVER_TREE_TYPEDEF_HPP

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
