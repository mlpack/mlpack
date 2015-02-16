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

  //! Return the sparse objective function matrix (sparseC).
  const arma::sp_mat& SparseC() const { return function.SparseC(); }

  //! Modify the sparse objective function matrix (sparseC).
  arma::sp_mat& SparseC() { return function.SparseC(); }

  //! Return the dense objective function matrix (denseC).
  const arma::mat& DenseC() const { return function.DenseC(); }

  //! Modify the dense objective function matrix (denseC).
  arma::mat& DenseC() { return function.DenseC(); }

  //! Return the vector of sparse A matrices (which correspond to the sparse
  // constraints).
  const std::vector<arma::sp_mat>& SparseA() const { return function.SparseA(); }

  //! Modify the veector of sparse A matrices (which correspond to the sparse
  // constraints).
  std::vector<arma::sp_mat>& SparseA() { return function.SparseA(); }

  //! Return the vector of dense A matrices (which correspond to the dense
  // constraints).
  const std::vector<arma::mat>& DenseA() const { return function.DenseA(); }

  //! Modify the veector of dense A matrices (which correspond to the dense
  // constraints).
  std::vector<arma::mat>& DenseA() { return function.DenseA(); }

  //! Return the vector of sparse B values.
  const arma::vec& SparseB() const { return function.SparseB(); }
  //! Modify the vector of sparse B values.
  arma::vec& SparseB() { return function.SparseB(); }

  //! Return the vector of dense B values.
  const arma::vec& DenseB() const { return function.DenseB(); }
  //! Modify the vector of dense B values.
  arma::vec& DenseB() { return function.DenseB(); }

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
