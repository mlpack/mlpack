/*
 * @file core/distributions/laplace_distribution.hpp
 * @author Zhihao Lou
 * @author Rohan Raj
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

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The multivariate Laplace distribution centered at 0 has pdf
 *
 * \f[
 * f(x|\theta) = \frac{1}{2 \theta}\exp\left(-\frac{\|x - \mu\|}{\theta}\right)
 * \f]
 *
 * given scale parameter \f$\theta\f$ and mean \f$\mu\f$.  This implementation
 * assumes a diagonal covariance, but a rewrite to support arbitrary
 * covariances is possible.
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
 * Note that because of the diagonal covariance restriction, much of the
 * algebra in the paper above becomes simplified, and the PDF takes roughly
 * the same form as the univariate case.
 */
template<typename MatType = arma::mat>
class LaplaceDistribution
{
 public:
  // Convenience typedefs.
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;

  /**
   * Default constructor, which creates a Laplace distribution with zero
   * dimension and zero scale parameter.
   */
  LaplaceDistribution() : scale(0) { }

  /**
   * Construct the Laplace distribution with the given scale and
   * dimensionality. The mean is initialized to zero.
   *
   * @param dimensionality Dimensionality of distribution.
   * @param scale Scale of distribution.
   */
  LaplaceDistribution(const size_t dimensionality, const double scale) :
      mean(zeros<VecType>(dimensionality)), scale(scale) { }

  /**
   * Construct the Laplace distribution with the given mean and scale
   * parameter.
   *
   * @param mean Mean of distribution.
   * @param scale Scale of distribution.
   */
  LaplaceDistribution(const VecType& mean, const double scale) :
      mean(mean), scale(scale) { }

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  /**
   * Return the probability of the given observation.
   *
   * @param observation Point to evaluate probability at.
   */
  ElemType Probability(const VecType& observation) const
  {
    return std::exp(LogProbability(observation));
  }

  /**
   * Evaluate probability density function of given observation.
   *
   * @param x List of observations.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const MatType& x, VecType& probabilities) const;

  /**
   * Return the log probability of the given observation.
   *
   * @param observation Point to evaluate logarithm of probability.
   */
  ElemType LogProbability(const VecType& observation) const;

  /**
   * Evaluate log probability density function of given observation.
   *
   * @param x List of observations.
   * @param logProbabilities Output probabilities for each input observation.
   */
  void LogProbability(const MatType& x, VecType& logProbabilities) const
  {
    logProbabilities.set_size(x.n_cols);
    for (size_t i = 0; i < x.n_cols; ++i)
    {
      logProbabilities(i) = LogProbability(x.unsafe_col(i));
    }
  }

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.  This is inlined for speed.
   *
   * @return Random observation from this Laplace distribution.
   */
  VecType Random() const
  {
    VecType result(mean.n_elem);
    result.randu();
    result = mean + scale *
        log(1.0 + 2.0 * sign(result - 0.5) * (result - 0.5));
    return result;
  }

  /**
   * Estimate the Laplace distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  [[deprecated("Will be removed in mlpack 5.0.0; use Train() instead")]]
  void Estimate(const MatType& observations);

  /**
   * Estimate the Laplace distribution from the given observations, taking into
   * account the probability of each observation actually being from this
   * distribution.
   */
  [[deprecated("Will be removed in mlpack 5.0.0; use Train() instead")]]
  void Estimate(const MatType& observations,
                const VecType& probabilities);

  /**
   * Estimate the Laplace distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Train(const MatType& observations);

  /**
   * Estimate the Laplace distribution from the given observations, taking into
   * account the probability of each observation actually being from this
   * distribution.
   */
  void Train(const MatType& observations,
             const VecType& probabilities);


  //! Return the mean.
  const VecType& Mean() const { return mean; }
  //! Modify the mean.
  VecType& Mean() { return mean; }

  //! Return the scale parameter.
  ElemType Scale() const { return scale; }
  //! Modify the scale parameter.
  ElemType& Scale() { return scale; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mean));
    ar(CEREAL_NVP(scale));
  }

 private:
  //! Mean of the distribution.
  VecType mean;
  //! Scale parameter of the distribution.
  ElemType scale;
};

} // namespace mlpack

// Include implementation.
#include "laplace_distribution_impl.hpp"

#endif
