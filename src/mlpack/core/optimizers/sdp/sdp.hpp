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
 */
class SDP
{
 public:

  SDP(const size_t n,
      const size_t numSparseConstraints,
      const size_t numDenseConstraints);

  size_t N() const { return n; }

  size_t N2bar() const { return n * (n + 1) / 2; }

  size_t NumSparseConstraints() const { return sparseB.n_elem; }

  size_t NumDenseConstraints() const { return denseB.n_elem; }

  size_t NumConstraints() const { return sparseB.n_elem + denseB.n_elem; }

  //! Return the sparse objective function matrix (sparseC).
  const arma::sp_mat& SparseC() const { return sparseC; }

  //! Modify the sparse objective function matrix (sparseC).
  arma::sp_mat& SparseC() {
    hasModifiedSparseObjective = true;
    return sparseC;
  }

  //! Return the dense objective function matrix (denseC).
  const arma::mat& DenseC() const { return denseC; }

  //! Modify the dense objective function matrix (denseC).
  arma::mat& DenseC() {
    hasModifiedDenseObjective = true;
    return denseC;
  }

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

  bool HasSparseObjective() const { return hasModifiedSparseObjective; }

  bool HasDenseObjective() const { return hasModifiedDenseObjective; }

  /**
   * Check whether or not the constraint matrices are linearly independent.
   *
   * Warning: possibly very expensive check
   */
  bool HasLinearlyIndependentConstraints() const;

 private:

  //! Dimension of the objective variable.
  size_t n;

  //! Sparse objective function matrix c.
  arma::sp_mat sparseC;

  //! Dense objective function matrix c.
  arma::mat denseC;

  //! If false, sparseC is zero
  bool hasModifiedSparseObjective;

  //! If false, denseC is zero
  bool hasModifiedDenseObjective;

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

#endif
