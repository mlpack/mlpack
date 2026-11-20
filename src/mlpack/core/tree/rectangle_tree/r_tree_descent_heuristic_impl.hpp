/**
 * @file core/tree/rectangle_tree/r_tree_descent_heuristic_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of RTreeDescentHeuristic, a class that chooses the best child
 * of a node in an R tree when inserting a new point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_tree_descent_heuristic.hpp"

namespace mlpack {

template<typename TreeType>
inline size_t RTreeDescentHeuristic::ChooseDescentNode(const TreeType* node,
                                                       const size_t point)
{
  // Convenience typedef.
  using ElemType = typename TreeType::ElemType;

  ElemType minScore = std::numeric_limits<ElemType>::max();
  int bestIndex = 0;
  ElemType bestVol = 0.0;

  for (size_t i = 0; i < node->NumChildren(); ++i)
  {
    ElemType v1 = 1.0;
    ElemType v2 = 1.0;
    for (size_t j = 0; j < node->Child(i).Bound().Dim(); ++j)
    {
      v1 *= node->Child(i).Bound()[j].Width();
      v2 *= node->Child(i).Bound()[j].Contains(node->Dataset().col(point)[j]) ?
          node->Child(i).Bound()[j].Width() :
          (node->Child(i).Bound()[j].Hi() < node->Dataset().col(point)[j] ?
              (node->Dataset().col(point)[j] - node->Child(i).Bound()[j].Lo()) :
              (node->Child(i).Bound()[j].Hi() - node->Dataset().col(point)[j]));
    }

    assert(v2 - v1 >= 0);

    if ((v2 - v1) < minScore)
    {
      minScore = v2 - v1;
      bestVol = v1;
      bestIndex = i;
    }
    else if ((v2 - v1) == minScore && v1 < bestVol)
    {
      bestVol = v1;
      bestIndex = i;
    }
  }

  return bestIndex;
}

template<typename TreeType>
inline size_t RTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* node,
    const TreeType* insertedNode)
{
  // Convenience typedef.
  using ElemType = typename TreeType::ElemType;

  ElemType minScore = std::numeric_limits<ElemType>::max();
  int bestIndex = 0;
  ElemType bestVol = 0.0;

  for (size_t i = 0; i < node->NumChildren(); ++i)
  {
    ElemType v1 = 1.0;
    ElemType v2 = 1.0;
    for (size_t j = 0; j < node->Child(i).Bound().Dim(); ++j)
    {
      v1 *= node->Child(i).Bound()[j].Width();
      v2 *= node->Child(i).Bound()[j].Contains(insertedNode->Bound()[j]) ?
          node->Child(i).Bound()[j].Width() :
          (insertedNode->Bound()[j].Contains(node->Child(i).Bound()[j]) ?
          insertedNode->Bound()[j].Width() :
          (insertedNode->Bound()[j].Lo() < node->Child(i).Bound()[j].Lo()
          ? (node->Child(i).Bound()[j].Hi() -
          insertedNode->Bound()[j].Lo()) : (insertedNode->Bound()[j].Hi() -
          node->Child(i).Bound()[j].Lo())));
    }

    assert(v2 - v1 >= 0);

    if ((v2 - v1) < minScore)
    {
      minScore = v2 - v1;
      bestVol = v1;
      bestIndex = i;
    }
    else if ((v2 - v1) == minScore && v1 < bestVol)
    {
      bestVol = v1;
      bestIndex = i;
    }
  }

  return bestIndex;
}

} // namespace mlpack

#endif
