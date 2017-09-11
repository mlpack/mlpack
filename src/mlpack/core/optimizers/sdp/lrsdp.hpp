/**
 * @file lrsdp.hpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

#include "lrsdp_function.hpp"

namespace mlpack {
namespace optimization {

/**
 * LRSDP is the implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).  This solver uses the augmented Lagrangian
 * optimizer to solve low-rank semidefinite programs.
 */
template <typename SDPType>
class LRSDP
{
 public:
  /**
   * Create an LRSDP to be optimized.  The solution will end up being a matrix
   * of size (rows) x (rank).  To construct each constraint and the objective
   * function, use the function SDP() in order to access the SDPType object
   * associated with this optimizer.
   *
   * @param numSparseConstraints Number of sparse constraints in the problem.
   * @param numDenseConstraints Number of dense constraints in the problem.
   * @param initialPoint Initial point of the optimization.
   * @param maxIterations Maximum number of iterations.
   */
  LRSDP(const size_t numSparseConstraints,
        const size_t numDenseConstraints,
        const arma::mat& initialPoint,
        const size_t maxIterations = 1000);

  /**
   * Create an LRSDP object with the given SDP problem to be solved, and the
   * given initial point.  Note that the SDP may be modified later by calling
   * SDP() to access the object.
   *
   * TODO: this is currently not implemented.
   *
   * @param sdp SDP to be solved.
   * @param initialPoint Initial point of the optimization.
   * @param maxIterations Maximum number of iterations.
   *
  LRSDP(const SDPType& sdp,
        const arma::mat& initialPoint,
        const size_t maxIterations = 1000);
   */

  /**
   * Optimize the LRSDP and return the final objective value.  The given
   * coordinates will be modified to contain the final solution.
   *
   * @param coordinates Starting coordinates for the optimization.
   */
  double Optimize(arma::mat& coordinates);

  //! Return the SDP that will be solved.
  const SDPType& SDP() const { return function.SDP(); }
  //! Modify the SDP that will be solved.
  SDPType& SDP() { return function.SDP(); }

  //! Return the function to be optimized.
  const LRSDPFunction<SDPType>& Function() const { return function; }
  //! Modify the function to be optimized.
  LRSDPFunction<SDPType>& Function() { return function; }

  //! Return the augmented Lagrangian object.
  const AugLagrangian& AugLag() const { return augLag; }
  //! Modify the augmented Lagrangian object.
  AugLagrangian& AugLag() { return augLag; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! Augmented lagrangian optimizer.
  AugLagrangian augLag;
  //! Function to optimize, which the AugLagrangian object holds.
  LRSDPFunction<SDPType> function;
  //! The maximum number of iterations for optimization.
  size_t maxIterations;
};

} // namespace optimization
} // namespace mlpack

// Include implementation
#include "lrsdp_impl.hpp"

#endif
