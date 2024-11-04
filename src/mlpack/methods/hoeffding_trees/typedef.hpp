/**
 * @file methods/hoeffding_trees/typedef.hpp
 * @author Ryan Curtin
 *
 * Useful typedefs.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP

#include "streaming_decision_tree.hpp"
#include "hoeffding_tree.hpp"

namespace mlpack {

using HoeffdingTreeType = StreamingDecisionTree<HoeffdingTree<>>;

} // namespace mlpack

#endif
