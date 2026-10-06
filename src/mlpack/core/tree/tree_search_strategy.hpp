/**
 * @file core/tree/tree_search_strategy.hpp
 * @author Ryan Curtin
 *
 * Defines the TreeSearchStrategy enum, which is used by different tree-based
 * algorithms like KNN, RangeSearch, and others.
 */
#ifndef MLPACK_CORE_TREE_TREE_SEARCH_STRATEGY_HPP
#define MLPACK_CORE_TREE_TREE_SEARCH_STRATEGY_HPP

namespace mlpack {

// TreeSearchStrategy represents the different dual-tree algorithm search
// strategies available.
enum TreeSearchStrategy
{
  NAIVE,
  SINGLE_TREE,
  DUAL_TREE,
  GREEDY_SINGLE_TREE
};

} // namespace mlpack

#endif
