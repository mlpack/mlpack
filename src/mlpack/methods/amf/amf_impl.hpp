namespace mlpack {
namespace amf {

/**
 * Construct the LMF object.
 */
template<typename InitializationRule,
         typename UpdateRule>
AMF<InitializationRule, UpdateRule>::AMF(
    const size_t maxIterations,
    const double minResidue,
    const InitializationRule initializeRule,
    const UpdateRule update) :
    maxIterations(maxIterations),
    minResidue(minResidue),
    initializeRule(initializeRule),
    update(update)
{
  if (minResidue < 0.0)
  {
    Log::Warn << "AMF::AMF(): minResidue must be a positive value ("
        << minResidue << " given). Setting to the default value of 1e-10.\n";
    this->minResidue = 1e-10;
  }
}

/**
 * Apply Latent Matrix Factorization to the provided matrix.
 *
 * @param V Input matrix to be factorized
 * @param W Basis matrix to be output
 * @param H Encoding matrix to output
 * @param r Rank r of the factorization
 */
template<typename InitializationRule,
         typename UpdateRule>
template<typename MatType>
void AMF<InitializationRule, UpdateRule>::Apply(
    const MatType& V,
    const size_t r,
    arma::mat& W,
    arma::mat& H) const
{
  const size_t n = V.n_rows;
  const size_t m = V.n_cols;

  // Initialize W and H.
  initializeRule.Initialize(V, r, W, H);

  Log::Info << "Initialized W and H." << std::endl;

  size_t iteration = 1;
  const size_t nm = n * m;
  double residue = minResidue;
  double normOld = 0;
  double norm = 0;
  arma::mat WH;

  while (residue >= minResidue && iteration != maxIterations)
  {
    // Update step.
    // Update the value of W and H based on the Update Rules provided
    update.WUpdate(V, W, H);
    update.HUpdate(V, W, H);

    // Calculate norm of WH after each iteration.
    WH = W * H;
    norm = sqrt(accu(WH % WH) / nm);

    if (iteration != 0)
    {
      residue = fabs(normOld - norm);
      residue /= normOld;
    }

    normOld = norm;

    iteration++;
  }

  Log::Info << "AMF converged to residue of " << sqrt(residue) << " in "
      << iteration << " iterations." << std::endl;
}

}; // namespace nmf
}; // namespace mlpack
