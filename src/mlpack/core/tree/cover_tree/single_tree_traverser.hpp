/**
 * @file core/tree/cover_tree/single_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the SingleTreeTraverser for the cover tree.  This implements a
 * single-tree breadth-first recursion with a pruning rule and a base case (two
 * point) rule.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP
#define MLPACK_CORE_TREE_COVER_TREE_SINGLE_TREE_TRAVERSER_HPP

#include <mlpack/prereqs.hpp>

#include "cover_tree.hpp"

namespace mlpack {

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename RuleType>
class CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    SingleTreeTraverser
{
 public:
  /**
   * Initialize the single tree traverser with the given rule.
   */
  SingleTreeTraverser(RuleType& rule);

  /**
   * Traverse the tree with the given point.
   *
   * @param queryIndex The index of the point in the query set which is used as
   *      the query point.
   * @param referenceNode The tree node to be traversed.
   */
  void Traverse(const size_t queryIndex, CoverTree& referenceNode);

  //! Get the number of prunes so far.
  size_t NumPrunes() const { return numPrunes; }
  //! Set the number of prunes (good for a reset to 0).
  size_t& NumPrunes() { return numPrunes; }

 private:
  size_t ScaleIndex(const int scale);

  // Reference to the rules with which the tree will be traversed.
  RuleType& rule;

  // The number of nodes which have been pruned during traversal.
  size_t numPrunes;

  // As we traverse, we will hold onto these MapEntry objects to track the
  // reference nodes that need further recursion.
  struct CoverTreeMapEntry
  {
    //! The node this entry refers to.
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>* node;
    //! The score of the node.
    double score;
    //! The base case evaluation.
    double baseCase;

    //! Comparison operator.
    bool operator<(const CoverTreeMapEntry& other) const
    {
      return (score < other.score);
    }
  };

  // Auxiliary data structures used during search.  We keep them at the class
  // level, because the memory allocated with them can be reused for each query
  // point.

  // The scale levels for each of the eight "hot" vectors of reference nodes.
  arma::ivec::fixed<8> hotScaleLevels;
  // Eight vectors of "hot" reference nodes; these will contain the next levels
  // to be recursed into.
  std::vector<CoverTreeMapEntry> hotScaleVectors[8];
  // A "cold" map of reference nodes with scales that are less than the smallest
  // scale contained in any of the hot vectors.
  std::map<int, std::vector<CoverTreeMapEntry>, std::greater<int>> mapQueue;
  // A vector of leaves (reference nodes with scale INT_MIN).
  std::vector<CoverTreeMapEntry> leafVector;
};

} // namespace mlpack

// Include implementation.
#include "single_tree_traverser_impl.hpp"

#endif
