/**
 * @file r_tree_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the RTreeSplit class, a class that splits the nodes of an R
 * tree, starting at a leaf node and moving upwards if necessary.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A Rectangle Tree has new points inserted at the bottom.  When these
 * nodes overflow, we split them, moving up the tree and splitting nodes
 * as necessary.
 */
template<typename TreeType>
class RTreeSplit
{
 public:
  //! Default constructor.
  RTreeSplit() { }

  //! Construct this with the specified node.
  RTreeSplit(const TreeType* /* node */) { }

  //! Create a copy of the other split.
  RTreeSplit(const TreeType& /* other */) { }

  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   */
  void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  bool SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);

 private:
  /**
   * Get the seeds for splitting a leaf node.
   */
  static void GetPointSeeds(const TreeType* tree, int& i, int& j);

  /**
   * Get the seeds for splitting a non-leaf node.
   */
  static void GetBoundSeeds(const TreeType* tree, int& i, int& j);

  /**
   * Assign points to the two new nodes.
   */
  static void AssignPointDestNode(TreeType* oldTree,
                                  TreeType* treeOne,
                                  TreeType* treeTwo,
                                  const int intI,
                                  const int intJ);

  /**
   * Assign nodes to the two new nodes.
   */
  static void AssignNodeDestNode(TreeType* oldTree,
                                 TreeType* treeOne,
                                 TreeType* treeTwo,
                                 const int intI,
                                 const int intJ);

  /**
   * Insert a node into another node.
   */
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);

 public:
  /**
   * Serialize the split.
   */
  template<typename Archive>
  void Serialize(Archive &, const unsigned int /* version */) { };

};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "r_tree_split_impl.hpp"

#endif
