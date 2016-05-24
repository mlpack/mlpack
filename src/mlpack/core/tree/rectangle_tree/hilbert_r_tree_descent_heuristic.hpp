/**
 * @file hilbert_r_tree_descent_heuristic.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of HilbertRTreeDescentHeuristic, a class that chooses the best child of a
 * node in an R tree when inserting a new point.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_DESCENT_HEURISTIC_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_DESCENT_HEURISTIC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

class HilbertRTreeDescentHeuristic
{
 public:
    template<typename TreeType>
  static size_t ChooseDescentNode(const TreeType* node, const arma::vec& point);

  template<typename TreeType>
  static size_t ChooseDescentNode(const TreeType* node,
                                  const TreeType* insertedNode);

};
} //  namespace tree
} //  namespace mlpack

#include "hilbert_r_tree_descent_heuristic_impl.hpp"

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_DESCENT_HEURISTIC_HPP
