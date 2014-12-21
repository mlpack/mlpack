/**
 * @file random_initializer.hpp
 * @author Nishant Mehta
 *
 * A very simple random dictionary initializer for SparseCoding; it is probably
 * not a very good choice.
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
#ifndef __MLPACK_METHODS_SPARSE_CODING_RANDOM_INITIALIZER_HPP
#define __MLPACK_METHODS_SPARSE_CODING_RANDOM_INITIALIZER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace sparse_coding {

/**
 * A DictionaryInitializer for use with the SparseCoding class.  This provides a
 * random, normally distributed dictionary, such that each atom has a norm of 1.
 */
class RandomInitializer
{
 public:
  /**
   * Initialize the dictionary randomly from a normal distribution, such that
   * each atom has a norm of 1.  This is simple enough to be included with the
   * definition.
   *
   * @param data Dataset to use for initialization.
   * @param atoms Number of atoms (columns) in the dictionary.
   * @param dictionary Dictionary to initialize.
   */
  static void Initialize(const arma::mat& data,
                         const size_t atoms,
                         arma::mat& dictionary)
  {
    // Create random dictionary.
    dictionary.randn(data.n_rows, atoms);

    // Normalize each atom.
    for (size_t j = 0; j < atoms; ++j)
      dictionary.col(j) /= norm(dictionary.col(j), 2);
  }
};

}; // namespace sparse_coding
}; // namespace mlpack

#endif
