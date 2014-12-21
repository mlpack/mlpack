/**
 * @file data_dependent_random_initializer.hpp
 * @author Nishant Mehta
 *
 * A sensible heuristic for initializing dictionaries for sparse coding.
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
#ifndef __MLPACK_METHODS_SPARSE_CODING_DATA_DEPENDENT_RANDOM_INITIALIZER_HPP
#define __MLPACK_METHODS_SPARSE_CODING_DATA_DEPENDENT_RANDOM_INITIALIZER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace sparse_coding {

/**
 * A data-dependent random dictionary initializer for SparseCoding.  This
 * creates random dictionary atoms by adding three random observations from the
 * data together, and then normalizing the atom.
 */
class DataDependentRandomInitializer
{
 public:
  /**
   * Initialize the dictionary by adding together three random observations from
   * the data, and then normalizing the atom.  This implementation is simple
   * enough to be included with the definition.
   *
   * @param data Dataset to initialize the dictionary with.
   * @param atoms Number of atoms in dictionary.
   * @param dictionary Dictionary to initialize.
   */
  static void Initialize(const arma::mat& data,
                         const size_t atoms,
                         arma::mat& dictionary)
  {
    // Set the size of the dictionary.
    dictionary.set_size(data.n_rows, atoms);

    // Create each atom.
    for (size_t i = 0; i < atoms; ++i)
    {
      // Add three atoms together.
      dictionary.col(i) = (data.col(math::RandInt(data.n_cols)) +
          data.col(math::RandInt(data.n_cols)) +
          data.col(math::RandInt(data.n_cols)));

      // Now normalize the atom.
      dictionary.col(i) /= norm(dictionary.col(i), 2);
    }
  }
};

}; // namespace sparse_coding
}; // namespace mlpack

#endif
