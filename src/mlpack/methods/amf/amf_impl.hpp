/**
 * @file amf_impl.hpp
 * @author Sumedh Ghaisas
 * @author Mohan Rajendran
 * @author Ryan Curtin
 *
 * Implementation of AMF class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
namespace mlpack {
namespace amf {

/**
 * Construct the AMF object.
 */
template<typename TerminationPolicyType,
         typename InitializationRuleType,
         typename UpdateRuleType>
AMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>::AMF(
    const TerminationPolicyType& terminationPolicy,
    const InitializationRuleType& initializationRule,
    const UpdateRuleType& update) :
    terminationPolicy(terminationPolicy),
    initializationRule(initializationRule),
    update(update)
{ }

/**
 * Apply Alternating Matrix Factorization to the provided matrix.
 *
 * @param V Input matrix to be factorized
 * @param W Basis matrix to be output
 * @param H Encoding matrix to output
 * @param r Rank r of the factorization
 */
template<typename TerminationPolicyType,
         typename InitializationRuleType,
         typename UpdateRuleType>
template<typename MatType>
double AMF<TerminationPolicyType, InitializationRuleType, UpdateRuleType>::
Apply(const MatType& V,
      const size_t r,
      arma::mat& W,
      arma::mat& H)
{
  // Initialize W and H.
  initializationRule.Initialize(V, r, W, H);

  Log::Info << "Initialized W and H." << std::endl;

  // initialize the update rule
  update.Initialize(V, r);
  // initialize the termination policy
  terminationPolicy.Initialize(V);

  // check if termination conditions are met
  while (!terminationPolicy.IsConverged(W, H))
  {
    // Update the values of W and H based on the update rules provided.
    update.WUpdate(V, W, H);
    update.HUpdate(V, W, H);
  }

  // get final residue and iteration count from termination policy
  const double residue = terminationPolicy.Index();
  const size_t iteration = terminationPolicy.Iteration();

  Log::Info << "AMF converged to residue of " << residue << " in "
      << iteration << " iterations." << std::endl;

  return residue;
}

} // namespace amf
} // namespace mlpack
