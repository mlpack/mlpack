/**
 * @file amf_impl.hpp
 * @author Sumedh Ghaisas
 * @author Mohan Rajendran
 * @author Ryan Curtin
 *
 * Implementation of AMF class.
 *
 * This file is part of MLPACK 1.0.10.
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

  update.Initialize(V, r);
  terminationPolicy.Initialize(V);

  while (!terminationPolicy.IsConverged(W, H))
  {
    // Update the values of W and H based on the update rules provided.
    update.WUpdate(V, W, H);
    update.HUpdate(V, W, H);
  }

  const double residue = terminationPolicy.Index();
  const size_t iteration = terminationPolicy.Iteration();

  Log::Info << "AMF converged to residue of " << residue << " in "
      << iteration << " iterations." << std::endl;

  return residue;
}

}; // namespace amf
}; // namespace mlpack
