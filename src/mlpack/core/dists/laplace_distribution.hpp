/*
 * @file laplace.hpp
 * @author Zhihao Lou
 *
 * Laplace (double exponential) distribution used in SA.
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

#ifndef __MLPACK_CORE_OPTIMIZER_SA_LAPLACE_DISTRIBUTION_HPP
#define __MLPACK_CORE_OPTIMIZER_SA_LAPLACE_DISTRIBUTION_HPP

namespace mlpack {
namespace distribution {

/**
 * The multivariate Laplace distribution centered at 0 has pdf
 *
 * \f[
 * f(x|\theta) = \frac{1}{2 \theta}\exp\left(-\frac{\|x - \mu\|}{\theta}\right)
 * \f]
 *
 * given scale parameter \f$\theta\f$ and mean \f$\mu\f$.  This implementation
 * assumes a diagonal covariance, but a rewrite to support arbitrary covariances
 * is possible.
 *
 * See the following paper for more information on the non-diagonal-covariance
 * Laplace distribution and estimation techniques:
 *
 * @code
 * @article{eltoft2006multivariate,
 *   title={{On the Multivariate Laplace Distribution}},
 *   author={Eltoft, Torbj\orn and Kim, Taesu and Lee, Te-Won},
 *   journal={IEEE Signal Processing Letters},
 *   volume={13},
 *   number={5},
 *   pages={300--304},
 *   year={2006}
 * }
 * @endcode
 *
 * Note that because of the diagonal covariance restriction, much of the algebra
 * in the paper above becomes simplified, and the PDF takes roughly the same
 * form as the univariate case.
 */
class LaplaceDistribution
{
 public:
  /**
   * Default constructor, which creates a Laplace distribution with zero
   * dimension and zero scale parameter.
   */
  LaplaceDistribution() : scale(0) { }

  /**
   * Construct the Laplace distribution with the given scale and dimensionality.
   * The mean is initialized to zero.
   *
   * @param dimensionality Dimensionality of distribution.
   * @param scale Scale of distribution.
   */
  LaplaceDistribution(const size_t dimensionality, const double scale) :
      mean(arma::zeros<arma::vec>(dimensionality)), scale(scale) { }

  /**
   * Construct the Laplace distribution with the given mean and scale parameter.
   *
   * @param mean Mean of distribution.
   * @param scale Scale of distribution.
   */
  LaplaceDistribution(const arma::vec& mean, const double scale) :
      mean(mean), scale(scale) { }

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  /**
   * Return the probability of the given observation.
   */
  double Probability(const arma::vec& observation) const;

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.  This is inlined for speed.
   *
   * @return Random observation from this Laplace distribution.
   */
  arma::vec Random() const
  {
    arma::vec result(mean.n_elem);
    result.randu();

    // Convert from uniform distribution to Laplace distribution.
    // arma::sign() does not exist in Armadillo < 3.920 so we have to do this
    // elementwise.
    for (size_t i = 0; i < result.n_elem; ++i)
    {
      if (result[i] < 0)
        result[i] = mean[i] + scale * result[i] * std::log(1 + 2.0 * (result[i]
            - 0.5));
      else
        result[i] = mean[i] - scale * result[i] * std::log(1 - 2.0 * (result[i]
            - 0.5));
    }

    return result;
  }

  /**
   * Estimate the Laplace distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Estimate(const arma::mat& observations);

  /**
   * Estimate the Laplace distribution from the given observations, taking into
   * account the probability of each observation actually being from this
   * distribution.
   */
  void Estimate(const arma::mat& observations,
                const arma::vec& probabilities);

  //! Return the mean.
  const arma::vec& Mean() const { return mean; }
  //! Modify the mean.
  arma::vec& Mean() { return mean; }

  //! Return the scale parameter.
  double Scale() const { return scale; }
  //! Modify the scale parameter.
  double& Scale() { return scale; }

  //! Return a string representation of the object.
  std::string ToString() const;

 private:
  //! Mean of the distribution.
  arma::vec mean;
  //! Scale parameter of the distribution.
  double scale;

};

}; // namespace distribution
}; // namespace mlpack

#endif
