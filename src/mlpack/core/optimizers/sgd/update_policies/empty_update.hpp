/**
 * @file empty_update.hpp
 * @author Arun Reddy
 *
 * Empty update for SGD
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_EMPTY_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_EMPTY_UPDATE_HPP

namespace mlpack {
namespace optimization {

/**
 * Empty update policy for SGD.
 *
 */
class EmptyUpdate {
 public:

  /**
   * Default constructor.
   */
  EmptyUpdate(){}

  /**
   * Function for runtime initialization.
   *
   * @param n_rows
   * @param n_cols
   */
  void Initialize(const size_t n_rows = 0,
            const size_t n_cols = 0)
  {
    in_rows = n_rows;
    in_cols = n_cols;
  }

  arma::mat Update(double stepSize,
              arma::mat gradient)
  {
    // performs the vanilla SGD update.
    return - stepSize * gradient;
  }

 private:

  size_t in_rows;

  size_t in_cols;
};

} // namespace optimization
} // namespace mlpack

#endif
