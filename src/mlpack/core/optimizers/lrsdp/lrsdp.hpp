/**
 * @file lrsdp.hpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_HPP
#define __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

class LRSDP
{
 public:
  LRSDP() { }

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
        const size_t rank,
        const size_t rows);

  /**
   * Optimize the LRSDP and return the final objective value.  The given
   * coordinates will be modified to contain the final solution.
   *
   * @param coordinates Starting coordinates for the optimization.
   */
  double Optimize(arma::mat& coordinates);
//                AugLagrangian<LRSDP> auglag);

  /**
   * Evaluate the objective function of the LRSDP (no constraints) at the given
   * coordinates.  This is used by AugLagrangian<LRSDP>.
   */
  double Evaluate(const arma::mat& coordinates) const;

  /**
   * Evaluate the gradient of the LRSDP (no constraints) at the given
   * coordinates.  This is used by AugLagrangian<LRSDP>.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const;

  /**
   * Evaluate a particular constraint of the LRSDP at the given coordinates.
   */
  double EvaluateConstraint(const size_t index,
                            const arma::mat& coordinates) const;

  /**
   * Evaluate the gradient of a particular constraint of the LRSDP at the given
   * coordinates.
   */
  void GradientConstraint(const size_t index,
                          const arma::mat& coordinates,
                          arma::mat& gradient) const;

  //! Get the number of constraints in the LRSDP.
  size_t NumConstraints() const { return b.n_elem; }

  //! Get the initial point of the LRSDP.
  const arma::mat& GetInitialPoint();

  //! Return the objective function matrix (C).
  const arma::mat& C() const { return c; }
  //! Modify the objective function matrix (C).
  arma::mat& C() { return c; }

  //! Return the vector of A matrices (which correspond to the constraints).
  const std::vector<arma::mat> A() const { return a; }
  //! Modify the veector of A matrices (which correspond to the constraints).
  std::vector<arma::mat>& A() { return a; }

  //! Return the vector of B values.
  const arma::vec& B() const { return b; }
  //! Modify the vector of B values.
  arma::vec& B() { return b; }

 private:
  // Should probably use sparse matrices for some of these.
  arma::mat c; // For objective function.
  std::vector<arma::mat> a; // A_i for each constraint.
  arma::vec b; // b_i for each constraint.

  // Initial point.
  arma::mat initialPoint;
};

}; // namespace optimization
}; // namespace mlpack

// Include implementation.
#include "lrsdp_impl.hpp"

#endif
