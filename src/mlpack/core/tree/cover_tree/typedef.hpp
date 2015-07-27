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

template<typename MetricType, typename StatisticType, typename MatType>
using StandardCoverTree = CoverTree<MetricType,
                                    StatisticType,
                                    MatType,
                                    FirstPointIsRoot>;

} // namespace tree
} // namespace mlpack

#endif
