/**
 * @file methods/reinforcement_learning/noise/ornstein_uhlenbeck.hpp
 * @author Tarek Elsayed
 *
 * This file is the implementation of OUNoise class.
 * Ornstein-Uhlenbeck process generates temporally correlated exploration,
 * and it effectively copes with physical control problems of inertia.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_NOISE_ORNSTEIN_UHLENBECK_HPP
#define MLPACK_METHODS_RL_NOISE_ORNSTEIN_UHLENBECK_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
class OUNoise
{
 public:
  /**
   * @param size The size of the noise vector.
   * @param mu The mean of the noise process.
   * @param theta The rate of mean reversion.
   * @param sigma The standard deviation of the noise.
   */
  OUNoise(int size,
          double mu = 0.0,
          double theta = 0.15,
          double sigma = 0.2) :
      mu(mu * ones<arma::colvec>(size)),
      theta(theta),
      sigma(sigma)
  {
    reset();
  }

  /**
   * Reset the internal state to the mean (mu).
   */
  void reset()
  {
    state = mu;
  }

  /**
   * Update the internal state and return it as a noise sample.
   *
   * @return Noise sample.
   */
  arma::colvec sample()
  {
    arma::colvec x = state;
    arma::colvec dx = theta * (mu - x) + sigma * randn<arma::colvec>(x.n_elem);
    state = x + dx;
    return state;
  }

 private:
  //! Locally-stored state of the noise process.
  arma::colvec state;

  //! Locally-stored mean of the noise process.
  arma::colvec mu;

  //! Locally-stored rate of mean reversion.
  double theta;

  //! Locally-stored standard deviation of the noise.
  double sigma;
};

} // namespace mlpack

#endif
