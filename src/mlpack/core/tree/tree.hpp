/**
 * @file core/tree/tree.hpp
 * @author Ryan Curtin
 *
 * Convenience include for all tree functionality in mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_TREE_HPP
#define MLPACK_CORE_TREE_TREE_HPP

#include "bounds.hpp"

#include "binary_space_tree.hpp"
#include "cosine_tree/cosine_tree.hpp"
#include "cover_tree.hpp"
#include "example_tree.hpp"
#include "octree.hpp"
#include "rectangle_tree.hpp"
#include "spill_tree.hpp"

#include "tree_traits.hpp"
#include "build_tree.hpp"

#include "statistic.hpp"
#include "traversal_info.hpp"
#include "greedy_single_tree_traverser.hpp"

#endif
