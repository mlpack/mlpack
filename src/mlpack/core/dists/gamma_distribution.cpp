/**
 * @file gamma_distribution.cpp
 * @author Yannis Mentekidis
 *
 * Implementation of the methods of GammaDistribution.
 */
#include "gamma_distribution.hpp"

#include <boost/math/special_functions/digamma.hpp>
#include "mlpack/core/boost_backport/trigamma.hpp"

using namespace mlpack;
using namespace mlpack::distribution;


// Returns true if computation converged.
inline bool GammaDistribution::converged(double aOld, double aNew)
{
  return (std::abs(aNew - aOld) / aNew) < tol;
}

// Fits an alpha and beta parameter to each dimension of the data.
void GammaDistribution::Train(const arma::mat& rdata)
{
  // Use pseudonym fittingSet regardless of if rdata was provided by user.
  const arma::mat& fittingSet = 
    rdata.n_elem == 0 ? referenceSet : rdata;

  // If fittingSet is empty, nothing to do.
  if (arma::size(fittingSet) == arma::size(arma::mat()))
    return;

  // Use boost's definitions of digamma and tgamma, and std::log.
  using boost::math::digamma;
  using boost::math::trigamma;
  using std::log;

  // Allocate space for alphas and betas (Assume independent rows).
  alpha.set_size(fittingSet.n_rows);
  beta.set_size(fittingSet.n_rows);

  // Treat each dimension (i.e. row) independently.
  for (size_t row = 0; row < fittingSet.n_rows; ++row)
  {
    // Calculate log(mean(x)) and mean(log(x)) required for fitting process.
    const double meanLogx = arma::mean(arma::log(fittingSet.row(row)));
    const double meanx = arma::mean(fittingSet.row(row));
    const double logMeanx = std::log(arma::mean(meanx));
    
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
      assert(aEst > 0);

    } while (! converged(aEst, aOld) );
    
    alpha(row) = aEst;
    beta(row) = meanx/aEst;
  }
  return;
}

