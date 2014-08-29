/**
 * @file first_point_is_root.hpp
 * @author Ryan Curtin
 *
 * A very simple policy for the cover tree; the first point in the dataset is
 * chosen as the root of the cover tree.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
#define __MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

/**
 * This class is meant to be used as a choice for the policy class
 * RootPointPolicy of the CoverTree class.  This policy determines which point
 * is used for the root node of the cover tree.  This particular implementation
 * simply chooses the first point in the dataset as the root.  A more complex
 * implementation might choose, for instance, the point with least maximum
 * distance to other points (the closest to the "middle").
 */
class FirstPointIsRoot
{
 public:
  /**
   * Return the point to be used as the root point of the cover tree.  This just
   * returns 0.
   */
  static size_t ChooseRoot(const arma::mat& /* dataset */) { return 0; }
};

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
