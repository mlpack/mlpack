/**
 * @file random_selection.hpp
 * @author Ryan Curtin
 *
 * Randomly select some points (with replacement) to use for the Nystroem
 * method.  Replacement is suboptimal, but for rank << number of points, this is
 * unlikely.
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
#ifndef __MLPACK_METHODS_NYSTROEM_METHOD_RANDOM_SELECTION_HPP
#define __MLPACK_METHODS_NYSTROEM_METHOD_RANDOM_SELECTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

class RandomSelection
{
 public:
  /**
   * Randomly select the specified number of points in the dataset.
   *
   * @param data Dataset to sample from.
   * @param m Number of points to select.
   * @return Indices of selected points from the dataset.
   */
  const static arma::Col<size_t> Select(const arma::mat& data, const size_t m)
  {
    arma::Col<size_t> selectedPoints(m);
    for (size_t i = 0; i < m; ++i)
      selectedPoints(i) = math::RandInt(0, data.n_cols);

    return selectedPoints;
  }
};

}; // namespace kernel
}; // namespace mlpack

#endif
