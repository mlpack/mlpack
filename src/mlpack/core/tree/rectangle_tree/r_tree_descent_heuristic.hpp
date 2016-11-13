/**
 * @file r_tree_descent_heuristic.hpp
 * @author Andrew Wells
 *
 * Definition of RTreeDescentHeuristic, a class that chooses the best child of a
 * node in an R tree when inserting a new point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_DESCENT_HEURISTIC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

/**
 * When descending a RectangleTree to insert a point, we need to have a way to
 * choose a child node when the point isn't enclosed by any of them.  This
 * heuristic is used to do so.
 */
class RTreeDescentHeuristic
{
 public:
  /**
   * Evaluate the node using a heuristic.  The heuristic guarantees two things:
   *
   * 1. If point is contained in (or on) the bound, the value returned is zero.
   * 2. If the point is not contained in (or on) the bound, the value returned
   *    is greater than zero.
   *
   * @param node The node that is being evaluated.
   * @param point The index of the point that is being inserted.
   */
  template<typename TreeType>
  static size_t ChooseDescentNode(const TreeType* node, const size_t point);

  /**
   * Evaluate the node using a heuristic.  The heuristic guarantees two things:
   *
   * 1. If point is contained in (or on) the bound, the value returned is zero.
   * 2. If the point is not contained in (or on) the bound, the value returned
   *    is greater than zero.
   *
   * @param node The node that is being evaluated.
   * @param insertedNode The node that is being inserted.
   */
  template<typename TreeType>
  static size_t ChooseDescentNode(const TreeType* node,
                                  const TreeType* insertedNode);
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "r_tree_descent_heuristic_impl.hpp"

#endif
