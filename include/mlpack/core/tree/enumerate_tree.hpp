/**
 * @file core/tree/enumerate_tree.hpp
 * @author Ivan (Jonan) Georgiev
 *
 * This file contains function that performs a simple depth-first walk on the tree
 * calling `Enter` and `Leave` methods of a provided walker.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_ENUMERATE_TREE_HPP
#define MLPACK_CORE_TREE_ENUMERATE_TREE_HPP

namespace mlpack {

// Actual implementation of the enumeration. The problem is the unified
// detection if we're on the root, because Enter and Leave expect the
// parent being passed.
template <class TreeType, class Walker>
void EnumerateTreeImpl(TreeType* tree, Walker& walker, bool root)
{
  if (root)
    walker.Enter(tree, (const TreeType*)nullptr);

  const size_t numChildren = tree->NumChildren();
  for (size_t i = 0; i < numChildren; ++i)
  {
    TreeType* child = tree->ChildPtr(i);
    walker.Enter(child, tree);
    EnumerateTreeImpl(child, walker, false);
    walker.Leave(child, tree);
  }

  if (root)
    walker.Leave(tree, (const TreeType*)nullptr);
}

/**
 * Traverses all nodes of the tree, including the inner ones. On each node
 * two methods of the `enumer` are called:
 *
 * Enter(TreeType* node, TreeType* parent);
 * Leave(TreeType* node, TreeType* parent);
 *
 * @param tree The tree to traverse.
 * @param walker An instance of custom class, receiver of the enumeration.
 */
template <class TreeType, class Walker>
inline void EnumerateTree(TreeType* tree, Walker& walker)
{
  EnumerateTreeImpl(tree, walker, true);
}

} // namespace mlpack


#endif // MLPACK_CORE_TREE_ENUMERATE_TREE_HPP
