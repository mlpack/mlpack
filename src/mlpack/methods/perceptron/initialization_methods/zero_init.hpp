/**
 * @file zero_init.hpp
 * @author Udit Saxena
 *
 * Implementation of ZeroInitialization policy for perceptrons.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP
#define _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_ZERO_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {

/**
 * This class is used to initialize the matrix weightVectors to zero.
 */
class ZeroInitialization
{
 public:
  ZeroInitialization() { }

  inline static void Initialize(arma::mat& W,
                                const size_t row,
                                const size_t col)
  {
    arma::mat tempWeights(row, col);
    tempWeights.fill(0.0);

    W = tempWeights;
  }
}; // class ZeroInitialization

}; // namespace perceptron
}; // namespace mlpack

#endif
