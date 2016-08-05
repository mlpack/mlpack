/**
 * @file r_plus_plus_tree_descent_heuristic.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of RPlusPlusTreeDescentHeuristic, a class that chooses the best
 * child of a node in an R++ tree when inserting a new point.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_DESCENT_HEURISTIC_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_DESCENT_HEURISTIC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

class RPlusPlusTreeDescentHeuristic
{
 public:
  /**
   * Evaluate the node using a heuristic. Returns the number of the node
   * with minimum largest Hilbert value is greater than the Hilbert value of
   * the point being inserted.
   *
   * @param node The node that is being evaluated.
   * @param point The number of the point that is being inserted.
   */
  template<typename TreeType>
  static size_t ChooseDescentNode(TreeType* node, const size_t point);

  /**
   * Evaluate the node using a heuristic. Returns the number of the node
   * with minimum largest Hilbert value is greater than the largest
   * Hilbert value of the point being inserted.
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

#include "r_plus_plus_tree_descent_heuristic_impl.hpp"

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_DESCENT_HEURISTIC_HPP
