/**
 * @file primal_dual.hpp
 * @author Stephen Tu
 *
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sdp/sdp.hpp>

namespace mlpack {
namespace optimization {

/**
 * Interface to a primal dual interior point solver.
 *
 * @tparam SDPType
 */
template <typename SDPType>
class PrimalDualSolver
{
 public:
  /**
   * Construct a new solver instance from a given SDP instance.
   * Uses a random, positive initialization point.
   *
   * @param sdp Initialized SDP to be solved.
   */
  PrimalDualSolver(const SDPType& sdp);

  /**
   * Construct a new solver instance from a given SDP instance.  Uses a random,
   * positive initialization point. Both initialX and initialZ need to be
   * positive definite matrices.
   *
   * @param sdp Initialized SDP to be solved.
   * @param initialX
   * @param initialYSparse
   * @param initialYDense
   * @param initialZ
   */
  PrimalDualSolver(const SDPType& sdp,
                   const arma::mat& initialX,
                   const arma::vec& initialYSparse,
                   const arma::vec& initialYDense,
                   const arma::mat& initialZ);

  /**
   * Invoke the optimization procedure, returning the converged values for the
   * primal and dual variables.
   *
   * @param X
   * @param ySparse
   * @param yDense
   * @param Z
   */
  double Optimize(arma::mat& X,
                  arma::vec& ySparse,
                  arma::vec& yDense,
                  arma::mat& Z);

  /**
   * Invoke the optimization procedure, and only return the primal variable.
   *
   * @param X
   */
  double Optimize(arma::mat& X)
  {
    arma::vec ysparse, ydense;
    arma::mat Z;
    return Optimize(X, ysparse, ydense, Z);
  }

  //! Return the underlying SDP instance.
  const SDPType& SDP() const { return sdp; }

  //! Modify tau. Typical values are 0.99.
  double& Tau() { return tau; }

  //! Modify the XZ tolerance.
  double& NormXzTol() { return normXzTol; }

  //! Modify the primal infeasibility tolerance.
  double& PrimalInfeasTol() { return primalInfeasTol; }

  //! Modify the dual infeasibility tolerance.
  double& DualInfeasTol() { return dualInfeasTol; }

  //! Modify the maximum number of iterations to run before converging.
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! The SDP problem instance to optimize.
  SDPType sdp;

  //! Starting point for X. Needs to be positive definite.
  arma::mat initialX;

  //! Starting lagrange multiplier for the sparse constraints.
  arma::vec initialYsparse;

  //! Starting lagrange multiplier for the sparse constraints.
  arma::vec initialYdense;

  //! Starting point for Z, the complementary slack variable. Needs to be
  //positive definite.
  arma::mat initialZ;

  //! The step size modulating factor. Needs to be a scalar in (0, 1).
  double tau;

  //! The tolerance on the norm of XZ required before terminating.
  double normXzTol;

  //! The tolerance required on the primal constraints required before
  //! terminating.
  double primalInfeasTol;

  //! The tolerance required on the dual constraint required before terminating.
  double dualInfeasTol;

  //! Maximum number of iterations to run. Set to 0 for no limit.
  size_t maxIterations;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "primal_dual_impl.hpp"

#endif
