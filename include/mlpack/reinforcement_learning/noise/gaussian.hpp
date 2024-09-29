/**
 * @file methods/reinforcement_learning/noise/gaussian.hpp
 * @author Tarek Elsayed
 *
 * This file is the implementation of GaussianNoise class.
 * Gaussian Noise is statistical noise with a normal distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_NOISE_GAUSSIAN_HPP
#define MLPACK_METHODS_RL_NOISE_GAUSSIAN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
class GaussianNoise
{
 public:
  /**
   * @param size The size of the noise vector.
   * @param mu The mean of the noise.
   * @param sigma The standard deviation of the noise.
   */
  GaussianNoise(const int size,
                const double mu = 0.0,
                const double sigma = 1.0) :
      size(size),
      mu(mu),
      sigma(sigma)
  { /* Nothing to do here */ }

  /**
   * Reset the internal state.
   */
  void reset()
  { /* Nothing to do here */ }

  /**
   * Return a noise sample.
   *
   * @return Noise sample.
   */
  arma::colvec sample()
  {
    return sigma * randn<arma::colvec>(size) + mu;
  }

 private:
  //! Locally-stored number of elements.
  const double size;

  //! Locally-stored mean of the noise process.
  const double mu;

  //! Locally-stored standard deviation of the noise.
  const double sigma;
};

} // namespace mlpack

#endif
