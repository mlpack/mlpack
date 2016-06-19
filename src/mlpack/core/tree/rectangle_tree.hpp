/**
 * @file rectangle_tree.hpp
 * @author Andrew Wells
 *
 * Include all the necessary files to use the Rectangle Type Trees (RTree,
 * RStarTree, XTree, and HilbertRTree).
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HPP

/* We include bounds.hpp since it gives us the necessary files.
 * However, we will not use the "ballbounds" option.
 */
#include "bounds.hpp"
#include "rectangle_tree/rectangle_tree.hpp"
#include "rectangle_tree/single_tree_traverser.hpp"
#include "rectangle_tree/single_tree_traverser_impl.hpp"
#include "rectangle_tree/dual_tree_traverser.hpp"
#include "rectangle_tree/dual_tree_traverser_impl.hpp"
#include "rectangle_tree/r_tree_split.hpp"
#include "rectangle_tree/r_star_tree_split.hpp"
#include "rectangle_tree/r_tree_descent_heuristic.hpp"
#include "rectangle_tree/r_star_tree_descent_heuristic.hpp"
#include "rectangle_tree/traits.hpp"
#include "rectangle_tree/x_tree_split.hpp"
#include "rectangle_tree/typedef.hpp"

#endif
