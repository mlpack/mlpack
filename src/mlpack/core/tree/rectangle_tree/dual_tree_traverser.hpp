/**
 * @file dual_tree_traverser.hpp
 * @author Andrew Wells
 *
 * A nested class of Rectangle Tree for traversing rectangle type trees
 * with a given set of rules which indicate the branches to prune and the
 * order in which to recurse.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_DUAL_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_DUAL_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>

#include "rectangle_tree.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename RuleType>
class RectangleTree<MetricType, StatisticType, MatType, SplitType,
                    DescentType, AuxiliaryInformationType>::DualTreeTraverser
{
 public:
  /**
   * Instantiate the dual-tree traverser with the given rule set.
   */
  DualTreeTraverser(RuleType& rule);

  /**
   * Traverse the two trees.  This does not reset the number of prunes.
   *
   * @param queryNode The query node to be traversed.
   * @param referenceNode The reference node to be traversed.
   * @param score The score of the current node combination.
   */
  void Traverse(RectangleTree& queryNode, RectangleTree& referenceNode);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of prunes.
  size_t& NumPrunes() { return numPrunes; }

  //! Get the number of visited combinations.
  size_t NumVisited() const { return numVisited; }
  //! Modify the number of visited combinations.
  size_t& NumVisited() { return numVisited; }

  //! Get the number of times a node combination was scored.
  size_t NumScores() const { return numScores; }
  //! Modify the number of times a node combination was scored.
  size_t& NumScores() { return numScores; }

  //! Get the number of times a base case was calculated.
  size_t NumBaseCases() const { return numBaseCases; }
  //! Modify the number of times a base case was calculated.
  size_t& NumBaseCases() { return numBaseCases; }

 private:

  // We use this struct and this function to make the sorting and scoring easy
  // and efficient:
  struct NodeAndScore
  {
    RectangleTree* node;
    double score;
    typename RuleType::TraversalInfoType travInfo;
  };

  static bool nodeComparator(const NodeAndScore& obj1,
                             const NodeAndScore& obj2)
  {
    return obj1.score < obj2.score;
  }

  //! Reference to the rules with which the trees will be traversed.
  RuleType& rule;

  //! The number of prunes.
  size_t numPrunes;

  //! The number of node combinations that have been visited during traversal.
  size_t numVisited;

  //! The number of times a node combination was scored.
  size_t numScores;

  //! The number of times a base case was calculated.
  size_t numBaseCases;

  //! Traversal information, held in the class so that it isn't continually
  //! being reallocated.
  typename RuleType::TraversalInfoType traversalInfo;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "dual_tree_traverser_impl.hpp"

#endif
