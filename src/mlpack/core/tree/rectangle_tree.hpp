/**
 * @file rectangle_tree.hpp
 * @author Andrew Wells
 * 
 * Include all the necessary filse to use the Rectangle Type Trees (RTree, RStarTree, XTree,
 * and HilbertRTree.)
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_HPP

/* we include bounds.hpp since it gives us the necessary files.
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

#endif
