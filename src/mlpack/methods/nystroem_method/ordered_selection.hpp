/**
 * @file ordered_selection.hpp
 * @author Ryan Curtin
 *
 * Select the first points of the dataset for use in the Nystroem method of
 * kernel matrix approximation. This is mostly for testing, but might have
 * other uses.
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
#ifndef __MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP
#define __MLPACK_METHODS_NYSTROEM_METHOD_ORDERED_SELECTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

class OrderedSelection
{
 public:
  /**
   * Select the specified number of points in the dataset.
   *
   * @param data Dataset to sample from.
   * @param m Number of points to select.
   * @return Indices of selected points from the dataset.
   */
  const static arma::Col<size_t> Select(const arma::mat& /* unused */,
                                        const size_t m)
  {
    // This generates [0 1 2 3 ... (m - 1)].
    return arma::linspace<arma::Col<size_t> >(0, m - 1, m);
  }
};

}; // namespace kernel
}; // namespace mlpack

#endif
