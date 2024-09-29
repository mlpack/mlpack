/**
 * @file core/tree/rectangle_tree/r_plus_plus_tree_descent_heuristic_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of RPlusPlusTreeDescentHeuristic, a class that chooses the
 * best child of a node in an R++ tree when inserting a new point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_plus_plus_tree_descent_heuristic.hpp"
#include "../hrectbound.hpp"

namespace mlpack {

template<typename TreeType>
size_t RPlusPlusTreeDescentHeuristic::ChooseDescentNode(
    TreeType* node, const size_t point)
{
  // Find the node whose maximum bounding rectangle contains the point.
  for (size_t bestIndex = 0; bestIndex < node->NumChildren(); bestIndex++)
  {
    if (node->Child(bestIndex).AuxiliaryInfo().OuterBound().Contains(
        node->Dataset().col(point)))
      return bestIndex;
  }

  // We should never reach this point.
  assert(false);

  return 0;
}

template<typename TreeType>
size_t RPlusPlusTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* /* node */, const TreeType* /* insertedNode */)
{
  // Should never be used.
  assert(false);

  return 0;
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_DESCENT_HEURISTIC_IMPL_HPP
