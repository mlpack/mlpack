/**
 * @file mvu.hpp
 * @author Ryan Curtin
 *
 * An implementation of Maximum Variance Unfolding.  This file defines an MVU
 * class as well as a class representing the objective function (a semidefinite
 * program) which MVU seeks to minimize.  Minimization is performed by the
 * Augmented Lagrangian optimizer (which in turn uses the L-BFGS optimizer).
 *
 * Note: this implementation of MVU does not work.  See #189.
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
#ifndef __MLPACK_METHODS_MVU_MVU_HPP
#define __MLPACK_METHODS_MVU_MVU_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace mvu {

/**
 * The MVU class is meant to provide a good abstraction for users.  The dataset
 * needs to be provided, as well as several parameters.
 *
 * - dataset
 * - new dimensionality
 */
class MVU
{
 public:
  MVU(const arma::mat& dataIn);

  void Unfold(const size_t newDim,
              const size_t numNeighbors,
              arma::mat& outputCoordinates);

 private:
  const arma::mat& data;
};

}; // namespace mvu
}; // namespace mlpack

#endif
