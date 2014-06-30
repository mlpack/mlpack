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
inline double RTreeDescentHeuristic::EvalNode(const HRectBound<>& bound, const arma::vec& point)
{
  double v1 = 1.0;
  double v2 = 1.0;
  for(size_t i = 0; i < bound.Dim(); i++) {
    v1 *= bound[i].Width();
    v2 *= bound[i].Contains(point[i]) ? bound[i].Width() : (bound[i].Hi() < point[i] ? (point[i] - bound[i].Lo()) :
      (bound[i].Hi() - point[i]));
  }
  assert(v2 - v1 >= 0);
  return v2 - v1;
}

}; // namespace tree
}; // namespace mlpack

#endif
