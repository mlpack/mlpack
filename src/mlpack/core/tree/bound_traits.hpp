/**
 * @file bound_traits.hpp
 * @author Ryan Curtin
 *
 * A class for template metaprogramming traits for bounds.
 *
 * This file is part of mlpack 2.0.0.
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
#ifndef __MLPACK_CORE_TREE_BOUND_TRAITS_HPP
#define __MLPACK_CORE_TREE_BOUND_TRAITS_HPP

namespace mlpack {
namespace bound {

/**
 * A class to obtain compile-time traits about BoundType classes.  If you are
 * writing your own BoundType class, you should make a template specialization
 * in order to set the values correctly.
 *
 * @see TreeTraits, KernelTraits
 */
template<typename BoundType>
struct BoundTraits
{
  //! If true, then the bounds for each dimension are tight.  If false, then the
  //! bounds for each dimension may be looser than the range of all points held
  //! in the bound.  This defaults to false.
  static const bool HasTightBounds = false;
};

} // namespace bound
} // namespace mlpack

#endif
