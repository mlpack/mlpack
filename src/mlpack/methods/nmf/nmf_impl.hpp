/**
 * @file nmf.cpp
 * @author Mohan Rajendran
 *
 * Implementation of NMF class to perform Non-Negative Matrix Factorization
 * on the given matrix.
 *
 * This file is part of MLPACK 1.0.3.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace mlpack {
namespace nmf {

/**
 * Construct the NMF object.
 */
template<typename InitializationRule,
         typename WUpdateRule,
         typename HUpdateRule>
NMF<InitializationRule, WUpdateRule, HUpdateRule>::NMF(
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
    Log::Warn << "NMF::NMF(): minResidue must be a positive value ("
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
void NMF<InitializationRule, WUpdateRule, HUpdateRule>::Apply(
    const arma::mat& V,
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
  double normOld;
  double norm;
  arma::mat WH;

  while (residue >= minResidue && iteration != maxIterations)
  {
    // Update step.
    // Update the value of W and H based on the Update Rules provided
    wUpdate.Update(V, W, H);
    hUpdate.Update(V, W, H);

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

  Log::Info << "NMF converged to residue of " << sqrt(residue) << " in "
      << iteration << " iterations." << std::endl;
}

}; // namespace nmf
}; // namespace mlpack
