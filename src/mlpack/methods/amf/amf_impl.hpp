/**
 * @file amf_impl.hpp
 * @author Sumedh Ghaisas
 */
namespace mlpack {
namespace amf {

/**
 * Construct the LMF object.
 */
template<typename TerminationPolicy,
         typename InitializationRule,
         typename UpdateRule>
AMF<TerminationPolicy, InitializationRule, UpdateRule>::AMF(
    const TerminationPolicy& t_policy,
    const InitializationRule& initializeRule,
    const UpdateRule& update) :
    t_policy(t_policy),
    initializeRule(initializeRule),
    update(update)
{ }

/**
 * Apply Latent Matrix Factorization to the provided matrix.
 *
 * @param V Input matrix to be factorized
 * @param W Basis matrix to be output
 * @param H Encoding matrix to output
 * @param r Rank r of the factorization
 */
template<typename TerminationPolicy,
         typename InitializationRule,
         typename UpdateRule>
template<typename MatType>
double AMF<TerminationPolicy, InitializationRule, UpdateRule>::Apply(
    const MatType& V,
    const size_t r,
    arma::mat& W,
    arma::mat& H)
{
  // Initialize W and H.
  initializeRule.Initialize(V, r, W, H);

  Log::Info << "Initialized W and H." << std::endl;

  arma::mat WH;

  update.Initialize(V, r);
  t_policy.Initialize(V);

  while (!t_policy.IsConverged())
  {
    // Update step.
    // Update the value of W and H based on the Update Rules provided
    update.WUpdate(V, W, H);
    update.HUpdate(V, W, H);

    t_policy.Step(W, H);
  }

  double residue = sqrt(t_policy.Index());
  size_t iteration = t_policy.Iteration();

  Log::Info << "AMF converged to residue of " << residue << " in "
      << iteration << " iterations." << std::endl;

  return residue;
}

}; // namespace nmf
}; // namespace mlpack
