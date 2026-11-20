/**
 * @file core/tree/rectangle_tree/r_plus_tree_descent_heuristic_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of HilbertRTreeDescentHeuristic, a class that chooses the best child
 * of a node in an R tree when inserting a new point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_plus_tree_descent_heuristic.hpp"
#include "../hrectbound.hpp"

namespace mlpack {

template<typename TreeType>
size_t RPlusTreeDescentHeuristic::ChooseDescentNode(TreeType* node,
                                                    const size_t point)
{
  using ElemType = typename TreeType::ElemType;
  size_t bestIndex = 0;
  bool success = true;

  // Try to find a node that contains the point.
  for (bestIndex = 0; bestIndex < node->NumChildren(); bestIndex++)
  {
    if (node->Child(bestIndex).Bound().Contains(
        node->Dataset().col(point)))
      return bestIndex;
  }

  // No one node contains the point. Try to enlarge a node in such a way, that
  // the resulting node do not overlap other nodes.
  for (bestIndex = 0; bestIndex < node->NumChildren(); bestIndex++)
  {
    HRectBound<EuclideanDistance, ElemType> bound =
        node->Child(bestIndex).Bound();
    bound |= node->Dataset().col(point);

    success = true;

    for (size_t j = 0; j < node->NumChildren(); ++j)
    {
      if (j == bestIndex)
        continue;
      success = false;
      // Two nodes overlap if and only if there are no dimension in which
      // they do not overlap each other.
      for (size_t k = 0; k < node->Bound().Dim(); ++k)
      {
        if (bound[k].Lo() >= node->Child(j).Bound()[k].Hi() ||
            node->Child(j).Bound()[k].Lo() >= bound[k].Hi())
        {
          // We found the dimension in which these nodes do not overlap
          // each other.
          success = true;
          break;
        }
      }
      if (!success) // These two nodes overlap each other.
        break;
    }
    if (success) // We found two nodes that do no overlap each other.
      break;
  }

  if (!success) // We could not find two nodes that do no overlap each other.
  {
    size_t depth = node->TreeDepth();

    // Create a new node into which we will insert the point.
    TreeType* tree = node;
    while (depth > 1)
    {
      TreeType* child = new TreeType(tree);

      tree->children[tree->NumChildren()++] = child;
      tree = child;
      depth--;
    }
    return node->NumChildren() - 1;
  }

  assert(bestIndex < node->NumChildren());

  return bestIndex;
}

template<typename TreeType>
size_t RPlusTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* /* node */, const TreeType* /*insertedNode */)
{
  // Should never be used.
  assert(false);

  return 0;
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_DESCENT_HEURISTIC_IMPL_HPP
