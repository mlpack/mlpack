/**
 * @file methods/amf/termination_policies/incomplete_incremental_termination.hpp
 * @author Sumedh Ghaisas
 *
 * Termination policy used in AMF (Alternating Matrix Factorization).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _MLPACK_METHODS_AMF_INCOMPLETE_INCREMENTAL_TERMINATION_HPP
#define _MLPACK_METHODS_AMF_INCOMPLETE_INCREMENTAL_TERMINATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class acts as a wrapper for basic termination policies to be used by
 * SVDIncompleteIncrementalLearning. This class calls the wrapped class functions
 * after every n calls to main class functions where n is the number of rows.
 *
 * @see AMF, SVDIncompleteIncrementalLearning
 */
template <class TerminationPolicy>
class IncompleteIncrementalTermination
{
 public:
  /**
   * Empty constructor
   *
   * @param tPolicy object of wrapped class.
   */
  IncompleteIncrementalTermination(
      TerminationPolicy tPolicy = TerminationPolicy()) :
      tPolicy(tPolicy), incrementalIndex(0), iteration(0)
  { /* Nothing to do here. */ }

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param V Input matrix to be factorized.
   */
  template<class MatType>
  void Initialize(const MatType& V)
  {
    tPolicy.Initialize(V);

    // Initialize incremental index to number of rows.
    incrementalIndex = V.n_rows;
    iteration = 0;
  }

  /**
   * Check if termination criterio is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  template<typename MatType>
  bool IsConverged(MatType& W, MatType& H)
  {
    // increment iteration count
    iteration++;

    // If the iteration count is a multiple of incremental index, return the
    // wrapped termination policy result.
    if (iteration % incrementalIndex == 0)
      return tPolicy.IsConverged(W, H);
    else
      return false;
  }

  //! Get current value of residue.
  double Index() const { return tPolicy.Index(); }

  //! Get current iteration count.
  const size_t& Iteration() const { return iteration; }

  //! Access maximum number of iterations.
  size_t MaxIterations() const { return tPolicy.MaxIterations(); }
  //! Modify maximum number of iterations.
  size_t& MaxIterations() { return tPolicy.MaxIterations(); }

  //! Access the wrapped termination policy.
  const TerminationPolicy& TPolicy() const { return tPolicy; }
  //! Modify the wrapped termination policy.
  TerminationPolicy& TPolicy() { return tPolicy; }

 private:
  //! Wrapped termination policy.
  TerminationPolicy tPolicy;

  //! Number of iterations after which wrapped class object will be called.
  size_t incrementalIndex;
  //! Current iteration count.
  size_t iteration;
}; // class IncompleteIncrementalTermination

} // namespace mlpack

#endif
