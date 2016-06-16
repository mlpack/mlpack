/**
 * @file rectangle_tree.hpp
 * @author Andrew Wells
 *
 * Include all the necessary files to use the Rectangle Type Trees (RTree,
 * RStarTree, XTree, and HilbertRTree).
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
