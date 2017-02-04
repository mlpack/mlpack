/**
 * @file hilbert_r_tree_descent_heuristic_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of HilbertRTreeDescentHeuristic, a class that chooses the best
 * child of a node in an R tree when inserting a new point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "hilbert_r_tree_descent_heuristic.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
size_t HilbertRTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* node,
    const size_t point)
{
  size_t bestIndex = 0;

  for (bestIndex = 0; bestIndex < node->NumChildren() - 1; bestIndex++)
    if (node->Child(bestIndex).AuxiliaryInfo().HilbertValue().
        CompareWithCachedPoint(node->Dataset().col(point)) > 0)
      break;

  return bestIndex;
}

template<typename TreeType>
size_t HilbertRTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* node,
    const TreeType* /* insertedNode */)
{
  size_t bestIndex = 0;

  for (bestIndex = 0; bestIndex < node->NumChildren() - 1; bestIndex++)
    if (node->Child(bestIndex).AuxiliaryInfo().HilbertValue().
        CompareWith(node, node->AuxiliaryInfo().HilbertValue()) > 0)
      break;

  return bestIndex;
}

} //  namespace tree
} //  namespace mlpack

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_DESCENT_HEURISTIC_IMPL_HPP
