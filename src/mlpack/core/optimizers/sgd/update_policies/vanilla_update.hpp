/**
 * @file vanilla_update.hpp
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
 * Vanilla update policy for SGD.
 *
 */
class VanillaUpdate {
 public:

  /**
   * Function for runtime initialization.
   *
   * @param n_rows
   * @param n_cols
   */
  void Initialize(const size_t n_rows,
                  const size_t n_cols)
  {/* Do Nothing */}

  arma::mat Update(double stepSize, arma::mat gradient)
  {
    // perform the vanilla SGD update.
    return - stepSize * gradient;
  }

};

} // namespace optimization
} // namespace mlpack

#endif
