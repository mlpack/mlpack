/**
 * @file dual_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * A dual-tree traverser for the cover tree.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_DUAL_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>
#include <queue>

namespace mlpack {
namespace tree {

//! Forward declaration of struct to be used for traversal.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
struct DualCoverTreeMapEntry;

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
template<typename RuleType>
class CoverTree<MetricType, RootPointPolicy, StatisticType>::DualTreeTraverser
{
 public:
  /**
   * Initialize the dual tree traverser with the given rule type.
   */
  DualTreeTraverser(RuleType& rule);

  /**
   * Traverse the two specified trees.
   *
   * @param queryNode Root of query tree.
   * @param referenceNode Root of reference tree.
   */
  void Traverse(CoverTree& queryNode, CoverTree& referenceNode);

  /**
   * Helper function for traversal of the two trees.
   */
  void Traverse(CoverTree& queryNode,
                std::map<int, std::vector<DualCoverTreeMapEntry<
                    MetricType, RootPointPolicy, StatisticType> > >&
                    referenceMap);

  //! Get the number of pruned nodes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of pruned nodes.
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! The instantiated rule set for pruning branches.
  RuleType& rule;

  //! The number of pruned nodes.
  size_t numPrunes;

  //! Prepare map for recursion.
  void PruneMap(CoverTree& candidateQueryNode,
                std::map<int, std::vector<DualCoverTreeMapEntry<
                    MetricType, RootPointPolicy, StatisticType> > >&
                    referenceMap,
                std::map<int, std::vector<DualCoverTreeMapEntry<
                    MetricType, RootPointPolicy, StatisticType> > >& childMap);

  void PruneMapForSelfChild(CoverTree& candidateQueryNode,
                            std::map<int, std::vector<DualCoverTreeMapEntry<
                                MetricType, RootPointPolicy, StatisticType> > >&
                                referenceMap);

  void ReferenceRecursion(CoverTree& queryNode,
                          std::map<int, std::vector<DualCoverTreeMapEntry<
                              MetricType, RootPointPolicy, StatisticType> > >&
                              referenceMap);
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "dual_tree_traverser_impl.hpp"

#endif
