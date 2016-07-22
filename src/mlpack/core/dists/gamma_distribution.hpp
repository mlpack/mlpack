/**
 * @file gamma_distribution.hpp
 * @author Yannis Mentekidis
 *
 * Implementation of a Gamma distribution of multidimensional data that fits
 * gamma parameters (alpha, beta) to data.
 * The fitting is done independently for each dataset dimension (row), based on
 * the assumption each dimension is fully indepeendent.
 *
 * Based on "Estimating a Gamma Distribution" by Thomas P. Minka:
 * research.microsoft.com/~minka/papers/minka-gamma.pdf
 */

#ifndef _MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_HPP
#define _MLPACK_CORE_DISTRIBUTIONS_GAMMA_DISTRIBUTION_HPP

#include <mlpack/core.hpp>
namespace mlpack{
namespace distribution{

/**
 * Class for fitting the Gamma Distribution to a dataset.
 */
class GammaDistribution
{
  public:
    /**
     * Empty constructor.
     */
    GammaDistribution() { /* Nothing to do. */ };

    /**
     * Destructor.
     */
    ~GammaDistribution() {};

    /**
     * This function trains (fits distribution parameters) to new data or the
     * dataset the object owns.
     *
     * @param rdata Reference data to fit parameters to.
     * @param tol Convergence tolerance. This is *not* an absolute measure:
     *    It will stop the approximation once the *change* in the value is 
     *    smaller than tol.
     */
    void Train(const arma::mat& rdata, const double tol = 1e-8);

    // Access to Gamma Distribution Parameters.
    /* Get alpha parameters of each dimension */
    arma::Col<double>& Alpha(void) { return alpha; };
    /* Get beta parameters of each dimension */
    arma::Col<double>& Beta(void) { return beta; };

  private:
    arma::Col<double> alpha; // Array of fitted alphas.
    arma::Col<double> beta; // Array of fitted betas.

    /**
     * This is a small function that returns true if the update of alpha is smaller
     * than the tolerance ratio.
     *
     * @param aOld old value of parameter we want to estimate (alpha in our case).
     * @param aNew new value of parameter (the value after 1 iteration from aOld).
     * @param tol Convergence tolerance. Relative measure (see documentation of
     * GammaDistribution::Train)
     */
    inline bool converged(const double aOld, 
                          const double aNew, 
                          const double tol);
};

} // namespace distributions.
} // namespace mlpack.

#endif
