/**
 * @file typedef.hpp
 * @author Ryan Curtin
 *
 * Useful typedefs.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_TYPEDEF_HPP

#include "streaming_decision_tree.hpp"
#include "hoeffding_tree.hpp"

namespace mlpack {
namespace tree {

typedef StreamingDecisionTree<HoeffdingTree<>> HoeffdingTreeType;

} // namespace tree
} // namespace mlpack

#endif
