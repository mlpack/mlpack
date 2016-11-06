/**
 * @file hilbert_r_tree_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the HilbertRTreeSplit class, a class that splits the nodes of an R
 * tree, starting at a leaf node and moving upwards if necessary.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * The splitting procedure for the Hilbert R tree.  The template parameter
 * splitOrder is the order of the splitting policy. The Hilbert R tree splits a
 * node on overflow, turning splitOrder nodes into (splitOrder + 1) nodes.
 *
 * @tparam splitOrder Number of nodes to split.
 */
template<size_t splitOrder = 2>
class HilbertRTreeSplit
{
 public:
  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   *
   * @param node The node that is being split.
   * @param relevels Not used.
   */
  template<typename TreeType>
  static void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   *
   * @param node The node that is being split.
   * @param relevels Not used.
   */
  template<typename TreeType>
  static bool SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);

 private:
  /**
   * Try to find splitOrder cooperating siblings in order to redistribute their
   * children evenly. Returns true on success.
   *
   * @param parent The parent of of the overflowing node.
   * @param iTree The number of the overflowing node.
   * @param firstSibling The first cooperating sibling.
   * @param lastSibling The last cooperating sibling.
   */
  template<typename TreeType>
  static bool FindCooperatingSiblings(TreeType* parent,
                                      const size_t iTree,
                                      size_t& firstSibling,
                                      size_t& lastSibling);

  /**
   * Redistribute the children of the cooperating siblings evenly among them.
   *
   * @param parent The parent of of the overflowing node.
   * @param firstSibling The first cooperating sibling.
   * @param lastSibling The last cooperating sibling.
   */
  template<typename TreeType>
  static void RedistributeNodesEvenly(const TreeType* parent,
                                      const size_t firstSibling,
                                      const size_t lastSibling);

  /**
   * Redistribute the points of the cooperating siblings evenly among them.
   *
   * @param parent The parent of of the overflowing node.
   * @param firstSibling The first cooperating sibling.
   * @param lastSibling The last cooperating sibling.
   */
  template<typename TreeType>
  static void RedistributePointsEvenly(TreeType* parent,
                                       const size_t firstSibling,
                                       const size_t lastSibling);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "hilbert_r_tree_split_impl.hpp"

#endif

