/**
 * @file core/tree/octree/single_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Definition of the single tree traverser for the octree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_OCTREE_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_OCTREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>
#include "octree.hpp"

namespace mlpack {

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename RuleType>
class Octree<DistanceType, StatisticType, MatType>::SingleTreeTraverser
{
 public:
  /**
   * Instantiate the traverser with the given rule set.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the reference tree with the given query point.  This does not
   * reset the number of pruned nodes.
   *
   * @param queryIndex Index of query point.
   * @param referenceNode Node in reference tree.
   */
  void Traverse(const size_t queryIndex, Octree& referenceNode);

  //! Get the number of pruned nodes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of pruned nodes.
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! The instantiated rule.
  RuleType& rule;
  //! The number of reference nodes that have been pruned.
  size_t numPrunes;
};

} // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif
