/**
 * @file nothing_initializer.hpp
 * @author Ryan Curtin
 *
 * An initializer for SparseCoding which does precisely nothing.  It is useful
 * for when you have an already defined dictionary and you plan on setting it
 * with SparseCoding::Dictionary().
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_METHODS_SPARSE_CODING_NOTHING_INITIALIZER_HPP
#define __MLPACK_METHODS_SPARSE_CODING_NOTHING_INITIALIZER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace sparse_coding {

/**
 * A DictionaryInitializer for SparseCoding which does not initialize anything;
 * it is useful for when the dictionary is already known and will be set with
 * SparseCoding::Dictionary().
 */
class NothingInitializer
{
 public:
  /**
   * This function does not initialize the dictionary.  This will cause problems
   * for SparseCoding if the dictionary is not set manually before running the
   * method.
   */
  static void Initialize(const arma::mat& /* data */,
                         const size_t /* atoms */,
                         arma::mat& /* dictionary */)
  {
    // Do nothing!
  }
};

}; // namespace sparse_coding
}; // namespace mlpack

#endif
