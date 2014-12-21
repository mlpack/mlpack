/**
 * @file lrsdp.hpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_HPP
#define __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

#include "lrsdp_function.hpp"

namespace mlpack {
namespace optimization {

/**
 * LRSDP is the implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).  This solver uses the augmented Lagrangian
 * optimizer to solve low-rank semidefinite programs.
 */
class LRSDP
{
 public:
  /**
   * Create an LRSDP to be optimized.  The solution will end up being a matrix
   * of size (rank) x (rows).  To construct each constraint and the objective
   * function, use the functions A(), B(), and C() to set them correctly.
   *
   * @param numConstraints Number of constraints in the problem.
   * @param rank Rank of the solution (<= rows).
   * @param rows Number of rows in the solution.
   */
  LRSDP(const size_t numConstraints,
        const arma::mat& initialPoint);

  /**
   * Create an LRSDP to be optimized, passing in an already-created
   * AugLagrangian object.  The given initial point should be set to the size
   * (rows) x (rank), where (rank) is the reduced rank of the problem.
   *
   * @param numConstraints Number of constraints in the problem.
   * @param initialPoint Initial point of the optimization.
   * @param auglag Pre-initialized AugLagrangian<LRSDP> object.
   */
  LRSDP(const size_t numConstraints,
        const arma::mat& initialPoint,
        AugLagrangian<LRSDPFunction>& augLagrangian);

  /**
   * Optimize the LRSDP and return the final objective value.  The given
   * coordinates will be modified to contain the final solution.
   *
   * @param coordinates Starting coordinates for the optimization.
   */
  double Optimize(arma::mat& coordinates);

  //! Return the objective function matrix (C).
  const arma::mat& C() const { return function.C(); }
  //! Modify the objective function matrix (C).
  arma::mat& C() { return function.C(); }

  //! Return the vector of A matrices (which correspond to the constraints).
  const std::vector<arma::mat>& A() const { return function.A(); }
  //! Modify the veector of A matrices (which correspond to the constraints).
  std::vector<arma::mat>& A() { return function.A(); }

  //! Return the vector of modes for the A matrices.
  const arma::uvec& AModes() const { return function.AModes(); }
  //! Modify the vector of modes for the A matrices.
  arma::uvec& AModes() { return function.AModes(); }

  //! Return the vector of B values.
  const arma::vec& B() const { return function.B(); }
  //! Modify the vector of B values.
  arma::vec& B() { return function.B(); }

  //! Return the function to be optimized.
  const LRSDPFunction& Function() const { return function; }
  //! Modify the function to be optimized.
  LRSDPFunction& Function() { return function; }

  //! Return the augmented Lagrangian object.
  const AugLagrangian<LRSDPFunction>& AugLag() const { return augLag; }
  //! Modify the augmented Lagrangian object.
  AugLagrangian<LRSDPFunction>& AugLag() { return augLag; }

  //! Return a string representation of the object.
  std::string ToString() const;

 private:
  //! Function to optimize, which the AugLagrangian object holds.
  LRSDPFunction function;

  //! The AugLagrangian object which will be used for optimization.
  AugLagrangian<LRSDPFunction> augLag;
};

}; // namespace optimization
}; // namespace mlpack

#endif
