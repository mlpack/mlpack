/*
 * @file laplace.hpp
 * @author Zhihao Lou
 *
 * Laplace (double exponential) distribution used in SA.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DISTRIBUTIONS_LAPLACE_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_LAPLACE_DISTRIBUTION_HPP

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
  double Probability(const arma::vec& observation) const
  {
    return exp(LogProbability(observation));
  }

  /**
   * Return the log probability of the given observation.
   */
  double LogProbability(const arma::vec& observation) const;

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
      if (result[i] < 0.5)
        result[i] = mean[i] + scale * std::log(1 + 2.0 * (result[i] - 0.5));
      else
        result[i] = mean[i] - scale * std::log(1 - 2.0 * (result[i] - 0.5));
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

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(mean, "mean");
    ar & data::CreateNVP(scale, "scale");
  }

 private:
  //! Mean of the distribution.
  arma::vec mean;
  //! Scale parameter of the distribution.
  double scale;

};

} // namespace distribution
} // namespace mlpack

#endif
