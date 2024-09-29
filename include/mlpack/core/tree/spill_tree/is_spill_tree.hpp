/**
 * @file core/tree/spill_tree/is_spill_tree.hpp
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

// Useful struct when specific behaviour for SpillTrees is required.
template<typename TreeType>
struct IsSpillTree
{
  static const bool value = false;
};

// Specialization for SpillTree.
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename HyperplaneDistanceType>
            class HyperplaneType,
         template<typename SplitDistanceType, typename SplitMatType>
            class SplitType>
struct IsSpillTree<SpillTree<DistanceType, StatisticType, MatType,
    HyperplaneType, SplitType>>
{
  static const bool value = true;
};

} // namespace mlpack

#endif
