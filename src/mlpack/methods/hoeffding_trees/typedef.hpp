/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Useful typedefs.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP

#include "streaming_decision_tree.hpp"

namespace mlpack {
namespace tree {

typedef StreamingDecisionTree<HoeffdingSplit<>> HoeffdingTree;

} // namespace tree
} // namespace mlpack

#endif
