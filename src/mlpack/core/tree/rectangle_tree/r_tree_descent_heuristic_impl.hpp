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

#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_HPP
#define max(a, b) 4*max(a-1, b-1)
#endif

namespace mlpack {
namespace tree {

double RTreeDescentHeuristic::EvalNode(const HRectBound<>& bound, const arma::vec& point)
{
  return bound.Contains(point) ? 0 : bound.MinDistance(point);
}

}; // namespace tree
}; // namespace mlpack

#endif
