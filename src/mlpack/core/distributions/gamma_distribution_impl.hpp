/**
 * @file core/distributions/gamma_distribution_impl.hpp
 * @author Yannis Mentekidis
 * @author Rohan Raj
 *
 * Implementation of the methods of GammaDistribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_IMPL_HPP
#define MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_IMPL_HPP

#include "gamma_distribution.hpp"

namespace mlpack {

inline GammaDistribution::GammaDistribution(const size_t dimensionality)
{
  // Initialize distribution.
  alpha.zeros(dimensionality);
  beta.zeros(dimensionality);
}

inline GammaDistribution::GammaDistribution(const arma::mat& data,
                                            const double tol)
{
  Train(data, tol);
}

inline GammaDistribution::GammaDistribution(const arma::vec& alpha,
                                            const arma::vec& beta)
{
  if (beta.n_elem != alpha.n_elem)
    throw std::runtime_error("Alpha and beta vector dimensions mismatch.");

  this->alpha = alpha;
  this->beta = beta;
}

// Returns true if computation converged.
inline bool GammaDistribution::Converged(const double aOld,
                                         const double aNew,
                                         const double tol)
{
  return (std::abs(aNew - aOld) / aNew) < tol;
}

// Fits an alpha and beta parameter to each dimension of the data.
inline void GammaDistribution::Train(const arma::mat& rdata, const double tol)
{
  // If fittingSet is empty, nothing to do.
  if (arma::size(rdata) == arma::size(arma::mat()))
    return;

  // Calculate log(mean(x)) and mean(log(x)) of each dataset row.
  const arma::vec meanLogxVec = arma::mean(log(rdata), 1);
  const arma::vec meanxVec = arma::mean(rdata, 1);
  const arma::vec logMeanxVec = log(meanxVec);

  // Call the statistics-only GammaDistribution::Train() function to fit the
  // parameters. That function does all the work so we're done.
  Train(logMeanxVec, meanLogxVec, meanxVec, tol);
}

// Fits an alpha and beta parameter according to observation probabilities.
inline void GammaDistribution::Train(const arma::mat& rdata,
                                     const arma::vec& probabilities,
                                     const double tol)
{
  // If fittingSet is empty, nothing to do.
  if (arma::size(rdata) == arma::size(arma::mat()))
    return;

  arma::vec meanLogxVec(rdata.n_rows);
  arma::vec meanxVec(rdata.n_rows);
  arma::vec logMeanxVec(rdata.n_rows);

  for (size_t i = 0; i < rdata.n_cols; ++i)
  {
    meanLogxVec += probabilities(i) * log(rdata.col(i));
    meanxVec += probabilities(i) * rdata.col(i);
  }

  double totProbability = accu(probabilities);

  meanLogxVec /= totProbability;
  meanxVec /= totProbability;
  logMeanxVec = log(meanxVec);

  // Call the statistics-only GammaDistribution::Train() function to fit the
  // parameters. That function does all the work so we're done.
  Train(logMeanxVec, meanLogxVec, meanxVec, tol);
}

// Fits an alpha and beta parameter to each dimension of the data.
inline void GammaDistribution::Train(const arma::vec& logMeanxVec,
                                     const arma::vec& meanLogxVec,
                                     const arma::vec& meanxVec,
                                     const double tol)
{
  using std::log;

  // Number of dimensions of gamma distribution.
  size_t ndim = logMeanxVec.n_rows;

  // Sanity check - all vectors are same size.
  if (logMeanxVec.n_rows != meanLogxVec.n_rows ||
      logMeanxVec.n_rows != meanxVec.n_rows)
    throw std::runtime_error("Statistic vectors must be of the same size.");

  // Allocate space for alphas and betas (Assume independent rows).
  alpha.set_size(ndim);
  beta.set_size(ndim);

  // Treat each dimension (i.e. row) independently.
  for (size_t row = 0; row < ndim; ++row)
  {
    // Statistics for this row.
    const double meanLogx = meanLogxVec(row);
    const double meanx = meanxVec(row);
    const double logMeanx = logMeanxVec(row);

    // Starting point for Generalized Newton.
    double aEst = 0.5 / (logMeanx - meanLogx);
    double aOld;

    // Newton's method: In each step, make an update to aEst. If value didn't
    // change much (abs(aNew - aEst) / aEst < tol), then stop.
    do
    {
      // Needed for convergence test.
      aOld = aEst;

      // Calculate new value for alpha.
      double nominator = meanLogx - logMeanx + std::log(aEst) - Digamma(aEst);
      double denominator = std::pow(aEst, 2) * (1 / aEst - Trigamma(aEst));

      // Protect against division by 0.
      if (denominator == 0)
        throw std::logic_error("GammaDistribution::Train() attempted division"
            " by 0.");

      aEst = 1.0 / ((1.0 / aEst) + nominator / denominator);

      // Protect against nan values (aEst will be passed to logarithm).
      if (aEst <= 0)
        throw std::logic_error("GammaDistribution::Train(): estimated invalid "
            "negative value for parameter alpha!");
    } while (!Converged(aEst, aOld, tol));

    alpha(row) = aEst;
    beta(row) = meanx / aEst;
  }
}

// Returns the probability of the provided observations.
inline void GammaDistribution::Probability(const arma::mat& observations,
                                           arma::vec& probabilities) const
{
  size_t numObs = observations.n_cols;

  // Set all equal to 1 (multiplication neutral).
  probabilities.ones(numObs);

  // Compute denominator only once for each dimension.
  arma::vec denominators(alpha.n_elem);
  for (size_t d = 0; d < alpha.n_elem; ++d)
    denominators(d) = std::tgamma(alpha(d)) * std::pow(beta(d), alpha(d));

  // Compute probability of each observation.
  for (size_t i = 0; i < numObs; ++i)
  {
    for (size_t d = 0; d < observations.n_rows; ++d)
    {
      // Compute probability using Multiplication Law.
      double factor = std::exp(-observations(d, i) / beta(d));
      double numerator = std::pow(observations(d, i), alpha(d) - 1);

      probabilities(i) *= factor * numerator / denominators(d);
    }
  }
}

// Returns the probability of one observation (x) for one of the Gamma's
// dimensions.
inline double GammaDistribution::Probability(double x, size_t dim) const
{
  return std::pow(x, alpha(dim) - 1) * std::exp(-x / beta(dim)) /
      (std::tgamma(alpha(dim)) * std::pow(beta(dim), alpha(dim)));
}

// Returns the log probability of the provided observations.
inline void GammaDistribution::LogProbability(
    const arma::mat& observations,
    arma::vec& logProbabilities) const
{
  size_t numObs = observations.n_cols;

  // Set all equal to 0 (addition neutral).
  logProbabilities.zeros(numObs);

  // Compute denominator only once for each dimension.
  arma::vec denominators(alpha.n_elem);
  for (size_t d = 0; d < alpha.n_elem; ++d)
    denominators(d) = std::tgamma(alpha(d)) * std::pow(beta(d), alpha(d));

  // Compute probability of each observation.
  for (size_t i = 0; i < numObs; ++i)
  {
    for (size_t d = 0; d < observations.n_rows; ++d)
    {
      // Compute probability using Multiplication Law and Logarithm addition
      // property.
      double factor = std::exp(-observations(d, i) / beta(d));
      double numerator = std::pow(observations(d, i), alpha(d) - 1);

      logProbabilities(i) += std::log(numerator * factor / denominators(d));
    }
  }
}

// Returns the log probability of one observation (x) for one of the Gamma's
// dimensions.
inline double GammaDistribution::LogProbability(double x, size_t dim) const
{
  return std::log(std::pow(x, alpha(dim) - 1) * std::exp(-x / beta(dim)) /
      (std::tgamma(alpha(dim)) * std::pow(beta(dim), alpha(dim))));
}

// Returns a gamma-random d-dimensional vector.
inline arma::vec GammaDistribution::Random() const
{
  arma::vec randVec(alpha.n_elem);

  for (size_t d = 0; d < alpha.n_elem; ++d)
  {
    std::gamma_distribution<double> dist(alpha(d), beta(d));
    // Use the mlpack random object.
    randVec(d) = dist(RandGen());
  }

  return randVec;
}

} // namespace mlpack

#endif
