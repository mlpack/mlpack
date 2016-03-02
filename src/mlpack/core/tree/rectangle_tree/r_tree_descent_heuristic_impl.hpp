/**
 * @file r_tree_descent_heuristic_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of RTreeDescentHeuristic, a class that chooses the best child
 * of a node in an R tree when inserting a new point.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_tree_descent_heuristic.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
inline size_t RTreeDescentHeuristic::ChooseDescentNode(const TreeType* node,
                                                       const arma::vec& point)
{
  double minScore = DBL_MAX;
  int bestIndex = 0;
  double bestVol = 0.0;

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    double v1 = 1.0;
    double v2 = 1.0;
    for (size_t j = 0; j < node->Children()[i]->Bound().Dim(); j++)
    {
      v1 *= node->Children()[i]->Bound()[j].Width();
      v2 *= node->Children()[i]->Bound()[j].Contains(point[j]) ?
          node->Children()[i]->Bound()[j].Width() :
          (node->Children()[i]->Bound()[j].Hi() < point[j] ?
              (point[j] - node->Children()[i]->Bound()[j].Lo()) :
              (node->Children()[i]->Bound()[j].Hi() - point[j]));
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
  double minScore = DBL_MAX;
  int bestIndex = 0;
  double bestVol = 0.0;

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    double v1 = 1.0;
    double v2 = 1.0;
    for (size_t j = 0; j < node->Children()[i]->Bound().Dim(); j++)
    {
      v1 *= node->Children()[i]->Bound()[j].Width();
      v2 *= node->Children()[i]->Bound()[j].Contains(insertedNode->Bound()[j]) ?
          node->Children()[i]->Bound()[j].Width() :
          (insertedNode->Bound()[j].Contains(node->Children()[i]->Bound()[j]) ?
          insertedNode->Bound()[j].Width() :
          (insertedNode->Bound()[j].Lo() < node->Children()[i]->Bound()[j].Lo()
          ? (node->Children()[i]->Bound()[j].Hi() -
          insertedNode->Bound()[j].Lo()) : (insertedNode->Bound()[j].Hi() -
          node->Children()[i]->Bound()[j].Lo())));
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

} // namespace tree
} // namespace mlpack

#endif
