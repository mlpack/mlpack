/**
 * @file nmf.hpp
 * @author Mohan Rajendran
 *
 * Defines the NMF class to perform Non-negative Matrix Factorization 
 * on the given matrix.
 */
#ifndef __MLPACK_METHODS_NMF_NMF_HPP
#define __MLPACK_METHODS_NMF_NMF_HPP

#include <mlpack/core.hpp>
#include "mdistupdate.hpp"
#include "randominit.hpp"

namespace mlpack {
namespace nmf {

/**
 * This class implements the NMF on the given matrix V. Non-negative Matrix 
 * Factorization decomposes V in the form \f$ V \approx WH \f$ where W is 
 * called the basis matrix and H is called the encoding matrix. V is taken 
 * to be of size n*m and the obtained W is n*r and H is r*m. The size r is 
 * called the rank of the factorization.
 * 
 * The implementation requires the supply of two templates. One for the update
 * rule for updating the W matrix during each iteration and another rule for
 * updating the H matrix during each iteration. This allows the user to
 * try out various update rules for performing the factorization.
 *
 * A simple example of how to run NMF is shown below.
 *
 * @code
 * extern arma::mat V; // Matrix that we want to perform NMF on.
 * size_t r = 10; // Rank of decomposition
 * arma::mat W; // Basis matrix
 * arma::mat H; // Encoding matrix
 * 
 * NMF<> nmf(); // Default options
 * nmf.Apply(V,W,H,r);
 * @endcode
 *
 * @tparam WUpdateRule The update rule for calculating W matrix at each 
 * iteration; @see MultiplicativeDistanceW for an example.
 * @tparam HUpdateRule The update rule for calculating H matrix at each
 * iteration; @see MultiplicativeDistanceH for an example.
 */
template<typename InitializeRule = RandomInitialization,
         typename WUpdateRule = MultiplicativeDistanceW,
         typename HUpdateRule = MultiplicativeDistanceH>
class NMF
{
 public:
  /**
   * Create the NMF object and (optionally) set the parameters which NMF will
   * run with. This implementation allows us to use different update rules for
   * the updation of the basis and encoding matrices over each iteration.
   * 
   * @param maxIterations Maximum number of iterations allowed before giving up
   * @param maxResidue The maximum root mean square of the difference between 
   *    two subsequent iteration of product WH at which to terminate iteration. 
   *    A low residual value denotes that subsequent iterationas are not 
   *    producing much different values of W and H. Once the difference goes 
   *    below the supplied value, the iteration terminates.
   * @param Initialize Optional Initialization object for initializing the
   *    W and H matrices
   * @param WUpdate Optional WUpdateRule object; for when the update rule for
   *    the W vector has states that it needs to store.
   * @param HUpdate Optional HUpdateRule object; for when the update rule for
   *    the H vector has states that it needs to store.
   */
  NMF(const size_t maxIterations = 10000,
      const double maxResidue = 1e-10,
      const InitializeRule Initialize = InitializeRule(),
      const WUpdateRule WUpdate = WUpdateRule(),
      const HUpdateRule HUpdate = HUpdateRule());

  /**
   * Apply the Non-Negative Matrix Factorization on the provided matrix.
   *
   * @param V Input matrix to be factorized
   * @param W Basis matrix to be output
   * @param H Encoding matrix to output
   * @param r Rank r of the factorization
   */
  void Apply(const arma::mat& V, arma::mat& W, arma::mat& H,
              size_t& r) const;

  private:
  //! The  maximum number of iterations allowed before giving up
  size_t maxIterations;
  //! The maximum residue below which iteration is considered converged
  double maxResidue;
  //! Instantiated W&H Initialization Rule
  InitializeRule Initialize;
  //! Instantiated W Update Rule
  WUpdateRule WUpdate;
  //! Instantiated H Update Rule
  HUpdateRule HUpdate;

}; // class NMF

}; // namespace nmf
}; // namespace mlpack

// Include implementation.
#include "nmf_impl.hpp"

#endif
