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
   * of size (rows) x (rank).  To construct each constraint and the objective
   * function, use the functions A(), B(), and C() to set them correctly.
   *
   * @param numConstraints Number of constraints in the problem.
   * @param initialPoint Initial point of the optimization.
   */
  LRSDP(const size_t numSparseConstraints,
        const size_t numDenseConstraints,
        const arma::mat& initialPoint);

  /**
   * Optimize the LRSDP and return the final objective value.  The given
   * coordinates will be modified to contain the final solution.
   *
   * @param coordinates Starting coordinates for the optimization.
   */
  double Optimize(arma::mat& coordinates);

  //! Return the sparse objective function matrix (C_sparse).
  const arma::sp_mat& C_sparse() const { return function.C_sparse(); }

  //! Modify the sparse objective function matrix (C_sparse).
  arma::sp_mat& C_sparse() { return function.C_sparse(); }

  //! Return the dense objective function matrix (C_dense).
  const arma::mat& C_dense() const { return function.C_dense(); }

  //! Modify the dense objective function matrix (C_dense).
  arma::mat& C_dense() { return function.C_dense(); }

  //! Return the vector of sparse A matrices (which correspond to the sparse
  // constraints).
  const std::vector<arma::sp_mat>& A_sparse() const { return function.A_sparse(); }

  //! Modify the veector of sparse A matrices (which correspond to the sparse
  // constraints).
  std::vector<arma::sp_mat>& A_sparse() { return function.A_sparse(); }

  //! Return the vector of dense A matrices (which correspond to the dense
  // constraints).
  const std::vector<arma::mat>& A_dense() const { return function.A_dense(); }

  //! Modify the veector of dense A matrices (which correspond to the dense
  // constraints).
  std::vector<arma::mat>& A_dense() { return function.A_dense(); }

  //! Return the vector of sparse B values.
  const arma::vec& B_sparse() const { return function.B_sparse(); }
  //! Modify the vector of sparse B values.
  arma::vec& B_sparse() { return function.B_sparse(); }

  //! Return the vector of dense B values.
  const arma::vec& B_dense() const { return function.B_dense(); }
  //! Modify the vector of dense B values.
  arma::vec& B_dense() { return function.B_dense(); }

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
