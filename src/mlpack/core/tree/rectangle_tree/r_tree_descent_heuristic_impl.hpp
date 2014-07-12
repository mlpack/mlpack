/**
 * @file r_tree_descent_heuristic_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of RTreeDescentHeuristic, a class that chooses the best child of a node in
 * an R tree when inserting a new point.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_tree_descent_heuristic.hpp"

namespace mlpack {
namespace tree {

// Return the increase in volume required when inserting point into bound.

template<typename TreeType>
inline size_t RTreeDescentHeuristic::ChooseDescentNode(const TreeType* node, const arma::vec& point)
{
  double minScore = DBL_MAX;
  int bestIndex = 0;

  for (size_t j = 0; j < node->NumChildren(); j++) {
    double v1 = 1.0;
    double v2 = 1.0;
    for (size_t i = 0; i < node->Child(j)->Bound().Dim(); i++) {
      v1 *= node->Child(j)->Bound()[i].Width();
      v2 *= node->Child(j)->Bound()[i].Contains(point[i]) ? node->Child(j)->Bound()[i].Width() : (node->Child(j)->Bound()[i].Hi() < point[i] ? (point[i] - node->Child(j)->Bound()[i].Lo()) :
              (node->Child(j)->Bound()[i].Hi() - point[i]));
    }
    assert(v2 - v1 >= 0);
    if (v2 - v1 < minScore) {
      minScore = v2 - v1;
      bestIndex = j;
    }
  }
  return bestIndex;
}

template<typename TreeType>
inline size_t RTreeDescentHeuristic::ChooseDescentNode(const TreeType* node, const TreeType* insertedNode)
{
  double minScore = DBL_MAX;
  int bestIndex = 0;

  for (size_t j = 0; j < node->NumChildren(); j++) {
    double v1 = 1.0;
    double v2 = 1.0;
    for (size_t i = 0; i < node->Child(j)->Bound().Dim(); i++) {
      v1 *= node->Child(j)->Bound()[i].Width();
      v2 *= node->Child(j)->Bound()[i].Contains(insertedNode->Bound()[i]) ? node->Child(j)->Bound()[i].Width() :
              (insertedNode->Bound()[i].Contains(node->Child(j)->Bound()[i]) ? insertedNode->Bound()[i].Width() : (insertedNode->Bound()[i].Lo() < node->Child(j)->Bound()[i].Lo() ? (node->Child(j)->Bound()[i].Hi() - insertedNode->Bound()[i].Lo()) : (insertedNode->Bound()[i].Hi() - node->Child(j)->Bound()[i].Lo())));
    }
    assert(v2 - v1 >= 0);
    if (v2 - v1 < minScore) {
      minScore = v2 - v1;
      bestIndex = j;
    }
  }
  return bestIndex;
}

}; // namespace tree
}; // namespace mlpack

#endif
