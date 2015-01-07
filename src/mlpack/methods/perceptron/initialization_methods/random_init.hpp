/**
 * @file random_init.hpp
 * @author Udit Saxena
 *
 * Random initialization for perceptron weights.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP
#define _MLPACK_METHOS_PERCEPTRON_INITIALIZATION_METHODS_RANDOM_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace perceptron {

/**
 * This class is used to initialize weights for the weightVectors matrix in a
 * random manner.
 */
class RandomInitialization
{
 public:
  RandomInitialization() { }

  inline static void Initialize(arma::mat& W,
                                const size_t row,
                                const size_t col)
  {
    W = arma::randu<arma::mat>(row, col);
  }
}; // class RandomInitialization

}; // namespace perceptron
}; // namespace mlpack

#endif
