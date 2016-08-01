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

namespace mlpack {
namespace distribution {

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
class GammaDistribution
{
  public:
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
    GammaDistribution(const arma::mat& data, const double tol = 1e-8);

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

    // Access to Gamma distribution parameters.

    //! Get the alpha parameter of the given dimension.
    double Alpha(const size_t dim) const { return alpha[dim]; }
    //! Modify the alpha parameter of the given dimension.
    double& Alpha(const size_t dim) { return alpha[dim]; }

    //! Get the beta parameter of the given dimension.
    double Beta(const size_t dim) const { return beta[dim]; }
    //! Modify the beta parameter of the given dimension.
    double& Beta(const size_t dim) { return beta[dim]; }

    //! Get the dimensionality of the distribution.
    size_t Dimensionality() const { return alpha.n_elem; }

  private:
    //! Array of fitted alphas.
    arma::vec alpha;
    //! Array of fitted betas.
    arma::vec beta;

    /**
     * This is a small function that returns true if the update of alpha is smaller
     * than the tolerance ratio.
     *
     * @param aOld old value of parameter we want to estimate (alpha in our case).
     * @param aNew new value of parameter (the value after 1 iteration from aOld).
     * @param tol Convergence tolerance. Relative measure (see documentation of
     * GammaDistribution::Train)
     */
    inline bool Converged(const double aOld,
                          const double aNew,
                          const double tol);
};

} // namespace distributions.
} // namespace mlpack.

#endif
