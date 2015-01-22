/**
 * @file sdp.hpp
 * @author Stephen Tu
 *
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SDP_SDP_HPP
#define __MLPACK_CORE_OPTIMIZERS_SDP_SDP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * Specify an SDP in primal form
 *
 *     min    dot(C, X)
 *     s.t.   dot(Ai, X) = bi, i=1,...,m, X >= 0
 *
 * @tparam ObjectiveMatrixType Should be either arma::mat or arma::sp_mat
 */
template <typename ObjectiveMatrixType>
class SDP
{
 public:

  typedef ObjectiveMatrixType objective_matrix_type;

  /**
   * Initialize this SDP to an empty state.
   */
  SDP();

  /**
   * Initialize this SDP to one which structurally has size n.
   *
   * @param n
   * @param numSparseConstraints
   * @param numDenseConstraints
   */
  SDP(const size_t n,
      const size_t numSparseConstraints,
      const size_t numDenseConstraints);

  size_t N() const { return c.n_rows; }

  size_t N2bar() const { return N() * (N() + 1) / 2; }

  size_t NumSparseConstraints() const { return sparseB.n_elem; }

  size_t NumDenseConstraints() const { return denseB.n_elem; }

  size_t NumConstraints() const { return sparseB.n_elem + denseB.n_elem; }

  //! Modify the sparse objective function matrix (sparseC).
  ObjectiveMatrixType& C() { return c; }

  //! Return the sparse objective function matrix (sparseC).
  const ObjectiveMatrixType& C() const { return c; }

  //! Return the vector of sparse A matrices (which correspond to the sparse
  // constraints).
  const std::vector<arma::sp_mat>& SparseA() const { return sparseA; }

  //! Modify the veector of sparse A matrices (which correspond to the sparse
  // constraints).
  std::vector<arma::sp_mat>& SparseA() { return sparseA; }

  //! Return the vector of dense A matrices (which correspond to the dense
  // constraints).
  const std::vector<arma::mat>& DenseA() const { return denseA; }

  //! Modify the veector of dense A matrices (which correspond to the dense
  // constraints).
  std::vector<arma::mat>& DenseA() { return denseA; }

  //! Return the vector of sparse B values.
  const arma::vec& SparseB() const { return sparseB; }
  //! Modify the vector of sparse B values.
  arma::vec& SparseB() { return sparseB; }

  //! Return the vector of dense B values.
  const arma::vec& DenseB() const { return denseB; }
  //! Modify the vector of dense B values.
  arma::vec& DenseB() { return denseB; }

  /**
   * Check whether or not the constraint matrices are linearly independent.
   *
   * Warning: possibly very expensive check
   */
  bool HasLinearlyIndependentConstraints() const;

 private:

  //! Objective function matrix c.
  ObjectiveMatrixType c;

  //! A_i for each sparse constraint.
  std::vector<arma::sp_mat> sparseA;
  //! b_i for each sparse constraint.
  arma::vec sparseB;

  //! A_i for each dense constraint.
  std::vector<arma::mat> denseA;
  //! b_i for each dense constraint.
  arma::vec denseB;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "sdp_impl.hpp"

#endif
