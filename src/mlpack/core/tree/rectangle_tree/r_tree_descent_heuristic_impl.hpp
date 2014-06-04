/**
 * @file r_tree_descent_heuristic_impl.hpp
 * @author Andrew Wells
 *
 * Definition of RTreeDescentHeuristic, a class that chooses the best child of a node in
 * an R tree when inserting a new point.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL.HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_IMPL.HPP

#include "r_tree_descent_heuristic.hpp"

namespace mlpack {
namespace tree {

/**
 * A binary space partitioning tree node is split into its left and right child.
 * The split is done in the dimension that has the maximum width. The points are
 * divided into two parts based on the mean in this dimension.
 */
template<typename BoundType, typename MatType = arma::mat>
double RTreeDescentHeuristic<BoundType, MatType>::EvalNode(const BoundType& bound, const arma::vec& point)
{
  return bound.contains(point) ? 0 : bound.minDistance(point);
}

}; // namespace tree
}; // namespace mlpack

#endif
