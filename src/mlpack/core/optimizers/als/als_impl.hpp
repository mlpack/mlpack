/**
 * @file nmf.cpp
 * @author Mohan Rajendran
 * @author Mudit Raj Gupta
 *
 * Implementation of NMF class to perform Non-Negative Matrix Factorization
 * on the given matrix.
 *
 */

namespace mlpack {
namespace als {

/**
 * Construct the NMF object.
 */
template<typename InitializationRule,
         typename WUpdateRule,
         typename HUpdateRule>
ALS<InitializationRule, WUpdateRule, HUpdateRule>::ALS(
    const size_t maxIterations,
    const double minResidue,
    const InitializationRule initializeRule,
    const WUpdateRule wUpdate,
    const HUpdateRule hUpdate) :
    maxIterations(maxIterations),
    minResidue(minResidue),
    initializeRule(initializeRule),
    wUpdate(wUpdate),
    hUpdate(hUpdate)
{
  if (minResidue < 0.0)
  {
    Log::Warn << "ALS::ALS(): minResidue must be a positive value ("
        << minResidue << " given). Setting to the default value of 1e-10.\n";
    this->minResidue = 1e-10;
  }
}

/**
 * Apply Non-Negative Matrix Factorization to the provided matrix.
 *
 * @param V Input matrix to be factorized
 * @param W Basis matrix to be output
 * @param H Encoding matrix to output
 * @param r Rank r of the factorization
 */
template<typename InitializationRule,
         typename WUpdateRule,
         typename HUpdateRule>
void ALS<InitializationRule, WUpdateRule, HUpdateRule>::Apply(
    const arma::sp_mat& V,
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
  arma::mat WH;
  double normNew = 0;
  while (residue >= minResidue && iteration != maxIterations)
  {
    // Update step.
    // Update the value of W and H based on the Update Rules provided
    wUpdate.Update(V, W, H);
    hUpdate.Update(V, W, H);
  
    // Calculate norm of WH after each iteration.
    WH = W * H;
    normNew = sqrt(accu(WH % WH) / nm);

    if (iteration != 0)
    {
      residue = fabs(normOld - normNew);
      residue /= normOld;
    }

    normOld = normNew;

    iteration++;
  }

  Log::Info << "ALS converged to residue of " << sqrt(residue) << " in "
      << iteration << " iterations." << std::endl;
}

}; // namespace als
}; // namespace mlpack
