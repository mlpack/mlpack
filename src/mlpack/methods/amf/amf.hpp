/**
 * @file amf.hpp
 * @author Sumedh Ghaisas
 */
#ifndef __MLPACK_METHODS_AMF_AMF_HPP
#define __MLPACK_METHODS_AMF_AMF_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/amf/update_rules/nmf_mult_dist.hpp>
#include <mlpack/methods/amf/init_rules/random_init.hpp>
#include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>

namespace mlpack {
namespace amf {

/**
 * This class implements the AMF on the given matrix V. Alternaing Matrix
 * Factorization decomposes V in the form \f$ V \approx WH \f$ where W is
 * called the basis matrix and H is called the encoding matrix. V is taken
 * to be of size n x m and the obtained W is n x r and H is r x m. The size r is
 * called the rank of the factorization.
 *
 * The implementation requires two template types; the first contains the
 * initialization rule for the W and H matrix and the other contains the update
 * rule to be used during each iteration.  This templatization allows the
 * user to try various update rules (including ones not supplied with MLPACK)
 * for factorization.
 *
 * A simple example of how to run AMF is shown below.
 *
 * @code
 * extern arma::mat V; // Matrix that we want to perform LMF on.
 * size_t r = 10; // Rank of decomposition
 * arma::mat W; // Basis matrix
 * arma::mat H; // Encoding matrix
 *
 * AMF<> amf; // Default options
 * amf.Apply(V, W, H, r);
 * @endcode
 *
 * @tparam InitializationRule The initialization rule for initializing W and H
 *     matrix.
 * @tparam UpdateRule The update rule for calculating W and H matrix at each
 *     iteration.
 *
 * @see NMF_MultiplicativeDistanceUpdate
 */
template<typename TerminationPolicy = SimpleResidueTermination,
         typename InitializationRule = RandomInitialization,
         typename UpdateRule = NMFMultiplicativeDistanceUpdate>
class AMF
{
 public:
  /**
   * Create the LMF object and (optionally) set the parameters which LMF will
   * run with.  The minimum residue refers to the root mean square of the
   * difference between two subsequent iterations of the product W * H.  A low
   * residue indicates that subsequent iterations are not producing much change
   * in W and H.  Once the residue goes below the specified minimum residue, the
   * algorithm terminates.
   *
   * @param maxIterations Maximum number of iterations allowed before giving up.
   *     A value of 0 indicates no limit.
   * @param minResidue The minimum allowed residue before the algorithm
   *     terminates.
   * @param Initialize Optional Initialization object for initializing the
   *     W and H matrices
   * @param Update Optional UpdateRule object; for when the update rule for
   *     the W and H vector has states that it needs to store
   */
  AMF(const TerminationPolicy& t_policy = TerminationPolicy(),
      const InitializationRule& initializeRule = InitializationRule(),
      const UpdateRule& update = UpdateRule());

  /**
   * Apply Latent Matrix Factorization to the provided matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be output.
   * @param H Encoding matrix to output.
   * @param r Rank r of the factorization.
   */
  template<typename MatType>
  double Apply(const MatType& V,
               const size_t r,
               arma::mat& W,
               arma::mat& H);

 private:
  //! termination policy
  TerminationPolicy t_policy;
  //! Instantiated initialization Rule.
  InitializationRule initializeRule;
  //! Instantiated update rule.
  UpdateRule update;

 public:
  //! Access the initialization rule.
  const InitializationRule& InitializeRule() const { return initializeRule; }
  //! Modify the initialization rule.
  InitializationRule& InitializeRule() { return initializeRule; }
  //! Access the update rule.
  const UpdateRule& Update() const { return update; }
  //! Modify the update rule.
  UpdateRule& Update() { return update; }
  //! Access the termination policy
  const TerminationPolicy& TPolicy() const { return t_policy; }
  //! Modify the termination policy
  TerminationPolicy& TPolicy() { return t_policy; }

}; // class AMF

}; // namespace amf
}; // namespace mlpack

// Include implementation.
#include "amf_impl.hpp"

#endif

