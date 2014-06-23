/**
 * @file nmf_als.hpp
 * @author Sumedh Ghaisas
 */
namespace mlpack {
namespace amf {

/**
 * Construct the LMF object.
 */
template<typename InitializationRule,
         typename UpdateRule>
AMF<InitializationRule, UpdateRule>::AMF(
    const size_t maxIterations,
    const double tolerance,
    const InitializationRule initializeRule,
    const UpdateRule update) :
    maxIterations(maxIterations),
    tolerance(tolerance),
    initializeRule(initializeRule),
    update(update)
{
  if (tolerance < 0.0 || tolerance > 1)
  {
    Log::Warn << "AMF::AMF(): tolerance must be a positive value in the range (0-1) but value "
        << tolerance << " is given. Setting to the default value of 1e-5.\n";
    this->tolerance = 1e-5;
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
double AMF<InitializationRule, UpdateRule>::Apply(
    const MatType& V,
    const size_t r,
    arma::mat& W,
    arma::mat& H)
{
  const size_t n = V.n_rows;
  const size_t m = V.n_cols;

  // Initialize W and H.
  initializeRule.Initialize(V, r, W, H);

  Log::Info << "Initialized W and H." << std::endl;

  size_t iteration = 1;
  const size_t nm = n * m;
  double residue = DBL_MIN;
  double oldResidue = DBL_MAX;
  double normOld = 0;
  double norm = 0;
  arma::mat WH;

  update.Initialize(V, r);

  while ((std::abs(oldResidue - residue) / oldResidue >= tolerance || iteration < 4) && iteration != maxIterations)
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
      oldResidue = residue;
      residue = fabs(normOld - norm);
      residue /= normOld;
    }

    normOld = norm;

    iteration++;
  }

  Log::Info << "AMF converged to residue of " << sqrt(residue) << " in "
      << iteration << " iterations." << std::endl;

  return residue;
}

}; // namespace nmf
}; // namespace mlpack

