/**
 * @file single_tree_traverser.hpp
 *
 * A nested class of VantagePointTree which traverses the entire tree with a
 * given set of rules which indicate the branches which can be pruned and the
 * order in which to recurse.  This traverser is a depth-first traverser.
 */
#ifndef MLPACK_CORE_TREE_VANTAGE_POINT_TREE_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_VANTAGE_POINT_TREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>

#include "vantage_point_tree.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename BoundMetricType, typename...> class BoundType,
         template<typename SplitBoundType, typename SplitMatType, size_t...>
             class SplitType>
template<typename RuleType>
class VantagePointTree<MetricType, StatisticType, MatType, BoundType,
                      SplitType>::SingleTreeTraverser
{
 public:
  /**
   * Instantiate the single tree traverser with the given rule set.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is being
   *     used as the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, VantagePointTree& referenceNode);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of prunes.
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  //! The number of nodes which have been pruned during traversal.
  size_t numPrunes;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif // MLPACK_CORE_TREE_VANTAGE_POINT_TREE_SINGLE_TREE_TRAVERSER_HPP

