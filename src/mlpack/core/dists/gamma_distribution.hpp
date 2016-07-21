/**
 * @file gamma_distribution.hpp
 * @author Yannis Mentekidis
 *
 * Implementation of a Gamma distribution of multidimensional data that fits
 * gamma parameters (alpha, beta) to data. Assumes each data dimension is
 * uncorrelated.
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
     * Empty constructor. Set tolerance to default.
     */
    GammaDistribution() : tol(1e-8)
    { /* Nothing to do. */ };

    /**
     * Constructor with reference data parameter.
     *
     * @param rdata Reference data for the object.
     */
    GammaDistribution(arma::mat& rdata) : referenceSet(rdata), tol(1e-8)
    { /* Nothing to do. */ };

    /**
     * Destructor,
     */
    ~GammaDistribution() {};

    /**
     * This function trains (fits distribution parameters) to new data or the
     * dataset the object owns.
     *
     * @param rdata Reference data to fit parameters to. If not specified,
     *    reference data will be used. Results are stored in the alpha and beta
     *    vectors (old results are removed).
     */
    void Train(const arma::mat& rdata = arma::mat());

    /* Access to referenceSet. */
    arma::mat& ReferenceSet(void) { return referenceSet; };
    /* Change referenceSet. */
    void ReferenceSet(arma::mat& rdata) { referenceSet = rdata; };

    /* Access to tolerance. */
    double Tol(void) const { return tol; };
    /* Change tolerance. */
    void Tol(double tolerance) { tol = tolerance; };

    // Access to Gamma Distribution Parameters.
    /* Get alpha parameters of each dimension */
    arma::Col<double>& Alpha(void) { return alpha; };
    /* Get beta parameters of each dimension */
    arma::Col<double>& Beta(void) { return beta; };

  private:
    arma::Col<double> alpha; // Array of fitted alphas.
    arma::Col<double> beta; // Array of fitted betas.
    arma::mat referenceSet; // Matrix of reference set.
    double tol; // Convergence tolerance.

    /**
     * This is a small function that returns true if the update of alpha is smaller
     * than the tolerance ratio.
     *
     * @param aOld old value of parameter we want to estimate (alpha in our case).
     * @param aNew new value of parameter (the value after 1 iteration from aOld).
     */
    inline bool converged(double aOld, double aNew);
};

} // namespace distributions.
} // namespace mlpack.

#endif
