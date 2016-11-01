/**
 * @file simple_residue_termination.hpp
 * @author Sumedh Ghaisas
 *
 * Termination policy used in AMF (Alternating Matrix Factorization).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This class implements a simple residue-based termination policy. The
 * termination decision depends on two factors: the value of the residue (the
 * difference between the norm of WH this iteration and the previous iteration),
 * and the number of iterations.  If the current value of residue drops below
 * the threshold or the number of iterations goes above the iteration limit,
 * IsConverged() will return true.  This class is meant for use with the AMF
 * (alternating matrix factorization) class.
 *
 * @see AMF
 */
class SimpleResidueTermination
{
 public:
  /**
   * Construct the SimpleResidueTermination object with the given minimum
   * residue (or the default) and the given maximum number of iterations (or the
   * default).  0 indicates no iteration limit.
   *
   * @param minResidue Minimum residue for termination.
   * @param maxIterations Maximum number of iterations.
   */
  SimpleResidueTermination(const double minResidue = 1e-5,
                           const size_t maxIterations = 10000)
      : minResidue(minResidue), maxIterations(maxIterations) { }

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param V Input matrix being factorized.
   */
  template<typename MatType>
  void Initialize(const MatType& V)
  {
    // Initialize the things we keep track of.
    residue = DBL_MAX;
    iteration = 1;
    nm = V.n_rows * V.n_cols;
    // Remove history.
    normOld = 0;
  }

  /**
   * Check if termination criterion is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // Calculate the norm and compute the residue, but do it by hand, so as to
    // avoid calculating (W*H), which may be very large.
    double norm = 0.0;
    for (size_t j = 0; j < H.n_cols; ++j)
      norm += arma::norm(W * H.col(j), "fro");
    residue = fabs(normOld - norm) / normOld;

    // Store the norm.
    normOld = norm;

    // Increment iteration count
    iteration++;
    Log::Info << "Iteration " << iteration << "; residue " << residue << ".\n";

    // Check if termination criterion is met.
    return (residue < minResidue || iteration > maxIterations);
  }

  //! Get current value of residue
  const double& Index() const { return residue; }

  //! Get current iteration count
  const size_t& Iteration() const { return iteration; }

  //! Access max iteration count
  const size_t& MaxIterations() const { return maxIterations; }
  size_t& MaxIterations() { return maxIterations; }

  //! Access minimum residue value
  const double& MinResidue() const { return minResidue; }
  double& MinResidue() { return minResidue; }

public:
  //! residue threshold
  double minResidue;
  //! iteration threshold
  size_t maxIterations;

  //! current value of residue
  double residue;
  //! current iteration count
  size_t iteration;
  //! norm of previous iteration
  double normOld;

  size_t nm;
}; // class SimpleResidueTermination

} // namespace amf
} // namespace mlpack


#endif // _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
