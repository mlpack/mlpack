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
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MVU_MVU_HPP
#define MLPACK_METHODS_MVU_MVU_HPP

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

} // namespace mvu
} // namespace mlpack

#endif
