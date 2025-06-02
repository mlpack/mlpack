/**
 * @file core/distributions/gamma_distribution.hpp
 * @author Yannis Mentekidis
 * @author Rohan Raj
 *
 * Implementation of a Gamma distribution of multidimensional data that fits
 * gamma parameters (alpha, beta) to data.
 * The fitting is done independently for each dataset dimension (row), based on
 * the assumption each dimension is fully independent.
 *
 * Based on "Estimating a Gamma Distribution" by Thomas P. Minka:
 * research.microsoft.com/~minka/papers/minka-gamma.pdf
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/math/digamma.hpp>
#include <mlpack/core/math/trigamma.hpp>

namespace mlpack {

/**
 * This class represents the Gamma distribution.  It supports training a Gamma
 * distribution on a given dataset and accessing the fitted alpha and beta
 * parameters.
 *
 * This class supports multidimensional Gamma distributions; however, it is
 * assumed that each dimension is independent; therefore, a multidimensional
 * Gamma distribution here may be seen as a set of independent
 * single-dimensional Gamma distributions---and the parameters are estimated
 * under this assumption.
 *
 * The estimation algorithm used can be found in the following paper:
 *
 * @code
 * @techreport{minka2002estimating,
 *   title={Estimating a {G}amma distribution},
 *   author={Minka, Thomas P.},
 *   institution={Microsoft Research},
 *   address={Cambridge, U.K.},
 *   year={2002}
 * }
 * @endcode
 */
template<typename MatType = arma::mat>
class GammaDistribution
{
 public:
  // Convenience typedefs.
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;

  /**
   * Construct the Gamma distribution with the given number of dimensions
   * (default 0); each parameter will be initialized to 0.
   *
   * @param dimensionality Number of dimensions.
   */
  GammaDistribution(const size_t dimensionality = 0);

  /**
   * Construct the Gamma distribution, training on the given parameters.
   *
   * @param data Data to train the distribution on.
   * @param tol Convergence tolerance. This is *not* an absolute measure:
   *    It will stop the approximation once the *change* in the value is
   *    smaller than tol.
   */
  GammaDistribution(const MatType& data,
                    const ElemType tol =
                        std::is_same_v<ElemType, float> ? 1e-4 : 1e-8);

  /**
   * Construct the Gamma distribution given two vectors alpha and beta.
   *
   * @param alpha The vector of alphas, one per dimension.
   * @param beta The vector of betas, one per dimension.
   */
  GammaDistribution(const VecType& alpha, const VecType& beta);

  /**
   * Destructor.
   */
  ~GammaDistribution() {}

  /**
   * This function trains (fits distribution parameters) to new data or the
   * dataset the object owns.
   *
   * @param rdata Reference data to fit parameters to.
   * @param tol Convergence tolerance. This is *not* an absolute measure:
   *    It will stop the approximation once the *change* in the value is
   *    smaller than tol.
   */
  void Train(const MatType& rdata,
             const ElemType tol =
                 std::is_same_v<ElemType, float> ? 1e-4 : 1e-8);

  /**
   * Fits an alpha and beta parameter according to observation probabilities.
   * This method is not yet implemented.
   *
   * @param observations The reference data, one observation per column.
   * @param probabilities The probability of each observation. One value per
   *     column of the observations matrix.
   * @param tol Convergence tolerance. This is *not* an absolute measure:
   *    It will stop the approximation once the *change* in the value is
   *    smaller than tol.
   */
  void Train(const MatType& observations,
             const VecType& probabilities,
             const ElemType tol =
                 std::is_same_v<ElemType, float> ? 1e-4 : 1e-8);

  /**
   * This function trains (fits distribution parameters) to a dataset with
   * pre-computed statistics logMeanx, meanLogx, meanx for each dimension.
   *
   * @param logMeanxVec Is each dimension's logarithm of the mean
   *     (log(mean(x))).
   * @param meanLogxVec Is each dimension's mean of logarithms
   *     (mean(log(x))).
   * @param meanxVec Is each dimension's mean (mean(x)).
   * @param tol Convergence tolerance. This is *not* an absolute measure:
   *    It will stop the approximation once the *change* in the value is
   *    smaller than tol.
   */
  void Train(const VecType& logMeanxVec,
             const VecType& meanLogxVec,
             const VecType& meanxVec,
             const ElemType tol =
                 std::is_same_v<ElemType, float> ? 1e-4 : 1e-8);

  /**
   * This function returns the probability of a group of observations.
   *
   * The probability of the value x is
   *
   * \f[
   * \frac{x^{(\alpha - 1)}}{\Gamma(\alpha) \beta^\alpha} e^{-\frac{x}{\beta}}
   * \f]
   *
   * for one dimension. This implementation assumes each dimension is
   * independent, so the product rule is used.
   *
   * @param observations Matrix of observations, one per column.
   * @param probabilities Column vector of probabilities, one per
   *     observation.
   */
  void Probability(const MatType& observations,
                   VecType& probabilities) const;

  /**
   * This function returns the probability of the given observation.
   *
   * @param x The observation to compute the probability of.
   */
  ElemType Probability(const VecType& x) const;

  /**
   * This is a shortcut to the Probability(arma::mat&, arma::vec&) function
   * for when we want to evaluate only the probability of one dimension of
   * the gamma.
   *
   * @param x The 1-dimensional observation.
   * @param dim The dimension for which to calculate the probability.
   */
  ElemType Probability(ElemType x, const size_t dim) const;

  /**
   * This function returns the logarithm of the probability of a group of
   * observations.
   *
   * The logarithm of the probability of a value x is
   *
   * \f[
   * \log(\frac{x^{(\alpha - 1)}}{\Gamma(\alpha) \beta^\alpha} e^
   * {-\frac{x}{\beta}})
   * \f]
   *
   * for one dimension. This implementation assumes each dimension is
   * independent, so the product rule is used.
   *
   * @param observations Matrix of observations, one per column.
   * @param logProbabilities Column vector of log probabilities, one per
   *     observation.
   */
  void LogProbability(const MatType& observations,
                      VecType& logProbabilities) const;

  /**
   * This function returns the log-probability of the given observation.
   *
   * @param x The observation to compute the log-probability of.
   */
  ElemType LogProbability(const VecType& x) const;

  /**
   * This function returns the logarithm of the probability of a single
   * observation.
   *
   * @param x The 1-dimensional observation.
   * @param dim The dimension for which to calculate the probability.
   */
  ElemType LogProbability(ElemType x, const size_t dim) const;

  /**
   * This function returns an observation of this distribution.
   */
  VecType Random() const;

  // Access to Gamma distribution parameters.

  //! Get the alpha parameter of the given dimension.
  ElemType Alpha(const size_t dim) const { return alpha[dim]; }
  //! Modify the alpha parameter of the given dimension.
  ElemType& Alpha(const size_t dim) { return alpha[dim]; }

  //! Get the beta parameter of the given dimension.
  ElemType Beta(const size_t dim) const { return beta[dim]; }
  //! Modify the beta parameter of the given dimension.
  ElemType& Beta(const size_t dim) { return beta[dim]; }

  //! Get the dimensionality of the distribution.
  size_t Dimensionality() const { return alpha.n_elem; }

 private:
  //! Array of fitted alphas.
  VecType alpha;
  //! Array of fitted betas.
  VecType beta;

  /**
   * This is a small function that returns true if the update of alpha is
   * smaller than the tolerance ratio.
   *
   * @param aOld Old value of parameter we want to estimate (alpha in our
   *     case).
   * @param aNew New value of parameter (the value after 1 iteration from
   *     aOld).
   * @param tol Convergence tolerance. Relative measure (see documentation of
   *     GammaDistribution::Train).
   */
  inline bool Converged(const ElemType aOld,
                        const ElemType aNew,
                        const ElemType tol);
};

} // namespace mlpack

// Include implementation.
#include "gamma_distribution_impl.hpp"

#endif
