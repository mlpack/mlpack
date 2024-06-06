/**
 * @file methods/amf/termination_policies/max_iteration_termination.hpp
 * @author Ryan Curtin
 *
 * A termination policy which only terminates when the maximum number of
 * iterations is reached.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_TERMINATION_POLICIES_MAX_ITERATION_TERMINATION_HPP
#define MLPACK_METHODS_AMF_TERMINATION_POLICIES_MAX_ITERATION_TERMINATION_HPP

namespace mlpack {

/**
 * This termination policy only terminates when the maximum number of iterations
 * has been reached.
 */
class MaxIterationTermination
{
 public:
  /**
   * Construct the termination policy with the given number of iterations
   * allowed (default 1000).  If maxIterations is 0, then termination will never
   * occur.
   *
   * @param maxIterations Maximum number of allowed iterations.
   */
  MaxIterationTermination(const size_t maxIterations = 1000) :
      maxIterations(maxIterations),
      iteration(0)
  {
    if (maxIterations == 0)
      Log::Warn << "MaxIterationTermination::MaxIterationTermination(): given "
          << "number of iterations is 0, so algorithm will never terminate!"
          << std::endl;
  }

  /**
   * Initialize for the given matrix V (there is nothing to do).
   */
  template<typename MatType>
  void Initialize(const MatType& /* V */) { }

  /**
   * Check if convergence has occurred.
   */
  template<typename MatType>
  bool IsConverged(const MatType& /* H */, const MatType& /* W */)
  {
    // Return true if we have performed the correct number of iterations.
    return (++iteration >= maxIterations);
  }

  //! Return something similar to the residue, which in this case is just the
  //! number of iterations left, since we don't have access to anything else.
  size_t Index() const
  {
    return (iteration > maxIterations) ? 0 : maxIterations - iteration;
  }

  //! Get the current iteration.
  size_t Iteration() const { return iteration; }
  //! Modify the current iteration.
  size_t& Iteration() { return iteration; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! The maximum number of allowed iterations.
  size_t maxIterations;
  //! The number of the current iteration.
  size_t iteration;
};

} // namespace mlpack

#endif
