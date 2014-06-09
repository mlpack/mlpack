/**
 * @file r_tree_descent_heuristic.hpp
 * @author Andrew Wells
 *
 * Definition of RTreeDescentHeuristic, a class that chooses the best child of a node in
 * an R tree when inserting a new point.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A binary space partitioning tree node is split into its left and right child.
 * The split is done in the dimension that has the maximum width. The points are
 * divided into two parts based on the mean in this dimension.
 */
template<typename MatType = arma::mat>
class RTreeDescentHueristic
{
 public:
  /**
   * Evaluate the node using a hueristic.  The heuristic guarantees two things:
   * 1.  If point is contained in (or on) bound, the value returned is zero.
   * 2.  If the point is not contained in (or on) bound, the value returned is greater than zero. 
   *
   * @param bound The bound used for the node that is being evaluated.
   * @param point The point that is being inserted.
   */
  static double EvalNode(const HRectBound& bound, const arma::vec& point);
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "r_tree_descent_heuristic_impl.hpp"

#endif
