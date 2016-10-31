/**
 * @file tree_traits.hpp
 * @author Ryan Curtin
 *
 * This file implements the basic, unspecialized TreeTraits class, which
 * provides information about tree types.  If you create a tree class, you
 * should specialize this class with the characteristics of your tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_TREE_TRAITS_HPP
#define MLPACK_CORE_TREE_TREE_TRAITS_HPP

namespace mlpack {
namespace tree {

/**
 * The TreeTraits class provides compile-time information on the characteristics
 * of a given tree type.  These include traits such as whether or not a node
 * knows the distance to its parent node, or whether or not the subspaces
 * represented by children can overlap.
 *
 * These traits can be used for static compile-time optimization:
 *
 * @code
 * // This if statement will be optimized out at compile time!
 * if (TreeTraits<TreeType>::HasOverlappingChildren == false)
 * {
 *   // Do a simpler computation because no children overlap.
 * }
 * else
 * {
 *   // Do the full, complex calculation.
 * }
 * @endcode
 *
 * The traits can also be used in conjunction with SFINAE to write specialized
 * versions of functions:
 *
 * @code
 * template<typename TreeType>
 * void Compute(TreeType& node,
 *              boost::enable_if<
 *                  TreeTraits<TreeType>::RearrangesDataset>::type*)
 * {
 *   // Computation where special dataset-rearranging tree constructor is
 *   // called.
 * }
 *
 * template<typename TreeType>
 * void Compute(TreeType& node,
 *              boost::enable_if<
 *                  !TreeTraits<TreeType>::RearrangesDataset>::type*)
 * {
 *   // Computation where normal tree constructor is called.
 * }
 * @endcode
 *
 * In those two examples, the boost::enable_if<> class takes a boolean template
 * parameter which allows that function to be called when the boolean is true.
 *
 * Each trait must be a static const value and not a function; only const values
 * can be used as template parameters (or constexprs can be used too).  By
 * default (the unspecialized implementation of TreeTraits), each parameter is
 * set to make as few assumptions about the tree as possible; so, even if
 * TreeTraits is not specialized for a particular tree type, tree-based
 * algorithms should still work.
 *
 * When you write your own tree, you must specialize the TreeTraits class to
 * your tree type and set the corresponding values appropriately.  See
 * mlpack/core/tree/binary_space_tree/traits.hpp for an example.
 */
template<typename TreeType>
class TreeTraits
{
 public:
  /**
   * This is true if the subspaces represented by the children of a node can
   * overlap.
   */
  static const bool HasOverlappingChildren = true;

  /**
   * This is true if a point can be included in more than one node.
   */
  static const bool HasDuplicatedPoints = false;

  /**
   * This is true if the first point of each node is the centroid of its bound.
   */
  static const bool FirstPointIsCentroid = false;

  /**
   * This is true if the points contained in the first child of a node
   * (Child(0)) are also contained in that node.
   */
  static const bool HasSelfChildren = false;

  /**
   * This is true if the tree rearranges points in the dataset when it is built.
   */
  static const bool RearrangesDataset = false;

  /**
   * This is true if the tree always has only two children.
   */
  static const bool BinaryTree = false;

  /**
   * This is true if the NumDescendants() method doesn't include duplicated
   * points.
   */
  static const bool UniqueNumDescendants = true;
};

} // namespace tree
} // namespace mlpack

#endif
