/**
 * @file core/tree/rectangle_tree/single_tree_traverser.hpp
 * @author Andrew Wells
 *
 * A nested class of Rectangle Tree for traversing rectangle type trees
 * with a given set of rules which indicate the branches to prune and the
 * order in which to recurse.  This is a depth-first traverser.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>

#include "rectangle_tree.hpp"

namespace mlpack {

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename RuleType>
class RectangleTree<DistanceType, StatisticType, MatType, SplitType,
                    DescentType, AuxiliaryInformationType>::SingleTreeTraverser
{
 public:
  /**
   * Instantiate the traverser with the given rule set.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is being
   *     used as the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, const RectangleTree& referenceNode);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of prunes.
  size_t& NumPrunes() { return numPrunes; }

 private:
  // We use this class and this function to make the sorting and scoring easy
  // and efficient:
  struct NodeAndScore
  {
    RectangleTree* node;
    double score;
  };

  static bool NodeComparator(const NodeAndScore& obj1, const NodeAndScore& obj2)
  {
    return obj1.score < obj2.score;
  }

  //! Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  //! The number of nodes which have been prenud during traversal.
  size_t numPrunes;
};

} // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif
