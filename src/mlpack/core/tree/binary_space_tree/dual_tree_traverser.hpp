/**
 * @file dual_tree_traverser.hpp
 * @author Ryan Curtin
 *
 * Defines the DualTreeTraverser for the BinarySpaceTree tree type.  This is a
 * nested class of BinarySpaceTree which traverses two trees in a depth-first
 * manner with a given set of rules which indicate the branches which can be
 * pruned and the order in which to recurse.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP

#include <mlpack/core.hpp>

#include "binary_space_tree.hpp"

namespace mlpack {
namespace tree {

template<typename BoundType, typename StatisticType, typename MatType>
template<typename RuleType>
class BinarySpaceTree<BoundType, StatisticType, MatType>::DualTreeTraverser
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
   */
  void Traverse(BinarySpaceTree& queryNode, BinarySpaceTree& referenceNode);

  //! Get the number of prunes.
  size_t NumPrunes() const { return numPrunes; }
  //! Modify the number of prunes.
  size_t& NumPrunes() { return numPrunes; }

 private:
  //! Reference to the rules with which the trees will be traversed.
  RuleType& rule;

  //! The number of nodes which have been pruned during traversal.
  size_t numPrunes;
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "dual_tree_traverser_impl.hpp"

#endif // __MLPACK_CORE_TREE_BINARY_SPACE_TREE_DUAL_TREE_TRAVERSER_HPP

