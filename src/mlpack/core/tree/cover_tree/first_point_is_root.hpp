/**
 * @file first_point_is_root.hpp
 * @author Ryan Curtin
 *
 * A very simple policy for the cover tree; the first point in the dataset is
 * chosen as the root of the cover tree.
 *
 * This file is part of mlpack 2.0.1.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
  template<typename MatType>
  static size_t ChooseRoot(const MatType& /* dataset */) { return 0; }
};

} // namespace tree
} // namespace mlpack

#endif // __MLPACK_CORE_TREE_FIRST_POINT_IS_ROOT_HPP
