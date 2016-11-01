/**
 * @file is_spill_tree.hpp
 *
 * Definition of IsSpillTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_IS_SPILL_TREE_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_IS_SPILL_TREE_HPP

#include "spill_tree.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

// Useful struct when specific behaviour for SpillTrees is required.
template<typename TreeType>
struct IsSpillTree
{
  static const bool value = false;
};

// Specialization for SpillTree.
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneMetricType>
            class HyperplaneType,
         template<typename SplitMetricType, typename SplitMatType>
            class SplitType>
struct IsSpillTree<tree::SpillTree<MetricType, StatisticType, MatType,
    HyperplaneType, SplitType>>
{
  static const bool value = true;
};

} // namespace tree
} // namespace mlpack

#endif
