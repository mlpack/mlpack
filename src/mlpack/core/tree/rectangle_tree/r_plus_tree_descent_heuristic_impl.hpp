/**
 * @file hilbert_r_tree_descent_heuristic_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of HilbertRTreeDescentHeuristic, a class that chooses the best child
 * of a node in an R tree when inserting a new point.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_plus_tree_descent_heuristic.hpp"
#include "../hrectbound.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
size_t RPlusTreeDescentHeuristic::
ChooseDescentNode(TreeType* node, const size_t point)
{
  typedef typename TreeType::ElemType ElemType;
  size_t bestIndex = 0;
  bool success;

  for (bestIndex = 0; bestIndex < node->NumChildren(); bestIndex++)
  {
    if (node->Children()[bestIndex]->Bound().Contains(node->Dataset().col(point)))
      return bestIndex;
  }

  for (bestIndex = 0; bestIndex < node->NumChildren(); bestIndex++)
  {
    bound::HRectBound<metric::EuclideanDistance, ElemType> bound =
        node->Children()[bestIndex]->Bound();
    bound |=  node->Dataset().col(point);

    success = true;

    for (size_t j = 0; j < node->NumChildren(); j++)
    {
      if (j == bestIndex)
        continue;
      success = false;
      for (size_t k = 0; k < node->Bound().Dim(); k++)
      {
        if (bound[k].Lo() >= node->Children()[j]->Bound()[k].Hi() ||
            node->Children()[j]->Bound()[k].Lo() >= bound[k].Hi())
        {
          success = true;
          break;
        }
      }
      if (!success)
        break;
    }
    if (success)
      break;
  }

  if (!success)
  {
    size_t depth = node->TreeDepth();

    TreeType* tree = node;
    while (depth > 1)
    {
      TreeType* child = new TreeType(node);

      tree->Children()[tree->NumChildren()++] = child;
      tree = child;
      depth--;
    }
    return node->NumChildren()-1;
  }

  assert(bestIndex < node->NumChildren());

  return bestIndex;
}

template<typename TreeType>
size_t RPlusTreeDescentHeuristic::
ChooseDescentNode(const TreeType* node, const TreeType* insertedNode)
{
  size_t bestIndex = 0;

  assert(false);

  return bestIndex;
}


} //  namespace tree
} //  namespace mlpack

#endif  //MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_DESCENT_HEURISTIC_IMPL_HPP
