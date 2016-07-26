/**
 * @file gamma_distribution.cpp
 * @author Yannis Mentekidis
 *
 * Implementation of the methods of GammaDistribution.
 */
#include "gamma_distribution.hpp"
#include <boost/math/special_functions/digamma.hpp>

using namespace mlpack;
using namespace mlpack::distribution;

GammaDistribution::GammaDistribution(const size_t dimensionality)
{
  // Initialize distribution.
  alpha.zeros(dimensionality);
  beta.zeros(dimensionality);
}

GammaDistribution::GammaDistribution(const arma::mat& data,
                                     const double tol)
{
  Train(data, tol);
}

// Returns true if computation converged.
inline bool GammaDistribution::Converged(const double aOld,
                                         const double aNew,
                                         const double tol)
{
  return (std::abs(aNew - aOld) / aNew) < tol;
}

// Fits an alpha and beta parameter to each dimension of the data.
void GammaDistribution::Train(const arma::mat& rdata, const double tol)
{
  // If fittingSet is empty, nothing to do.
  if (arma::size(rdata) == arma::size(arma::mat()))
    return;

  // Use boost's definitions of digamma and tgamma, and std::log.
  using boost::math::digamma;
  using boost::math::trigamma;
  using std::log;

  // Allocate temporary space for alphas and betas (Assume independent rows).
  // We need temporary vectors since we are calling Train(logMeanx, meanLogx,
  // meanx) which modifies the object's own alphas and betas.
  arma::vec tempAlpha(rdata.n_rows);
  arma::vec tempBeta(rdata.n_rows);

  // Calculate log(mean(x)) and mean(log(x)) of each dataset row.
  const arma::vec meanLogxVec = arma::mean(arma::log(rdata), 1);
  const arma::vec meanxVec = arma::mean(rdata, 1);
  const arma::vec logMeanxVec = arma::log(meanxVec);

  // Treat each dimension (i.e. row) independently.
  for (size_t row = 0; row < rdata.n_rows; ++row)
  {
    // Statistics for this Vrow.
    const double meanLogx = meanLogxVec(row);
    const double meanx = meanxVec(row);
    const double logMeanx = logMeanxVec(row);

    // Use the statistics-only function to fit this dimension.
    Train(logMeanx, meanLogx, meanx, tol);

    // The function above modifies the state of this object. Get the parameters
    // it fit. (This is not very good design...).
    tempAlpha(row) = alpha(0);
    tempBeta(row) = beta(0);
  }
  alpha = tempAlpha;
  beta = tempBeta;

}

// Fits an alpha and beta parameter to each dimension of the data.
void GammaDistribution::Train(const double logMeanx, const double meanLogx,
                              const double meanx,
                              const double tol)
{
  // Use boost's definitions of digamma and tgamma, and std::log.
  using boost::math::digamma;
  using boost::math::trigamma;
  using std::log;

  // Allocate space for alphas and betas (Assume independent rows).
  alpha.set_size(1);
  beta.set_size(1);


  // Starting point for Generalized Newton.
  double aEst = 0.5 / (logMeanx - meanLogx);
  double aOld;

  // Newton's method: In each step, make an update to aEst. If value didn't
  // change much (abs(aNew - aEst)/aEst < tol), then stop.
  do
  {
    // Needed for convergence test.
    aOld = aEst;

    // Calculate new value for alpha.
    double nominator = meanLogx - logMeanx + log(aEst) - digamma(aEst);
    double denominator = pow(aEst, 2) * (1 / aEst - trigamma(aEst));
    assert (denominator != 0); // Protect against division by 0.
    aEst = 1.0 / ((1.0 / aEst) + nominator / denominator);

    // Protect against nan values (aEst will be passed to logarithm).
    if (aEst <= 0)
      throw std::logic_error("GammaDistribution::Train(): estimated invalid "
          "negative value for parameter alpha!");

  } while (!Converged(aEst, aOld, tol));

  alpha(0) = aEst;
  beta(0) = meanx / aEst;
}
