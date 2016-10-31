/**
 * @file binary_space_tree.hpp
 * @author Ryan Curtin
 *
 * Include all the necessary files to use the BinarySpaceTree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_HPP

#include <mlpack/core.hpp>
#include "bounds.hpp"
#include "binary_space_tree/midpoint_split.hpp"
#include "binary_space_tree/mean_split.hpp"
#include "binary_space_tree/vantage_point_split.hpp"
#include "binary_space_tree/rp_tree_max_split.hpp"
#include "binary_space_tree/rp_tree_mean_split.hpp"
#include "binary_space_tree/ub_tree_split.hpp"
#include "binary_space_tree/binary_space_tree.hpp"
#include "binary_space_tree/single_tree_traverser.hpp"
#include "binary_space_tree/single_tree_traverser_impl.hpp"
#include "binary_space_tree/dual_tree_traverser.hpp"
#include "binary_space_tree/dual_tree_traverser_impl.hpp"
#include "binary_space_tree/breadth_first_dual_tree_traverser.hpp"
#include "binary_space_tree/breadth_first_dual_tree_traverser_impl.hpp"
#include "binary_space_tree/traits.hpp"
#include "binary_space_tree/typedef.hpp"

#endif
