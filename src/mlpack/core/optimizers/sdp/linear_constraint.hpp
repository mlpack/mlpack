/**
 * @file linear_constraint.hpp
 * @author Evan Patterson
 *
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SDP_LINEAR_CONSTRAINT_HPP
#define __MLPACK_CORE_OPTIMIZERS_SDP_LINEAR_CONSTRAINT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * Abstract base class for linear constraints in an SDP.
 *
 * This class computes affine constraints of the form
 *     f(X) == b
 * where f is a linear functional.
 *
 * Any linear functional f can be represented as f(X) = Tr(A*X) for some matrix
 * A. The standard concrete implementations DenseMatrixConstraint and
 * SparseMatrixConstraint specify f in terms of A.
 */
class LinearConstraint
{
public:
  /**
   * Evaluate the SDP constraint f(X) - b.
   */
  virtual arma::vec Evaluate(const arma::mat& x) = 0;

  /**
   * Evaluate the SDPLR constraint f(RR^T) - b, where R is low rank.
   *
   * The default implementation simply computes X = RR^T, then calls
   * Evaluate(). This is not always the most efficient way to do the
   * computation.
   */
  virtual arma::vec EvaluateLR(const arma::mat& r) {
    return Evaluate(r * trans(r));
  }

  /**
   * Evaluate the gradient of the SDPLR Lagrangian, namely
   *    2*S*R
   * where
   *    S = -\sum_{i=1}^m y_i A_i
   * Here A_i is the matrix defining the i-th linear functional.
   */
  virtual arma::mat GradientLR(const arma::mat& r, const arma::vec& y) = 0;
};

/**
 * Linear constraints defined by a matrix A:
 *
 *     f(X) = Tr(A*X)
 *
 * @tparam MatrixType Should be either arma::mat or arma::sp_mat.
 */
template <typename MatrixType>
class MatrixConstraint : public virtual LinearConstraint
{
public:
  MatrixConstraint();
  MatrixConstraint(size_t n);

  // LinearConstraint interface.
  virtual arma::vec Evaluate(const arma::mat& x);
  virtual arma::vec EvaluateLR(const arma::mat& r);
  virtual arma::mat GradientLR(const arma::mat& r, const arma::vec& y);

  //! Return the vector of A matrices.
  const std::vector<MatrixType>& A() const { return _a; }

  //! Modify the vector of A matrices.
  std::vector<MatrixType>& A() { return _a; }

  //! Return the vector of b values.
  const arma::vec& B() const { return _b; }
  
  //! Modify the vector of b values.
  arma::vec& B() { return _b; }

private:
  std::vector<MatrixType> _a;
  arma::vec _b;
};

typedef MatrixConstraint<arma::mat> DenseMatrixConstraint;
typedef MatrixConstraint<arma::sp_mat> SparseMatrixConstraint;

/**
 * Linear constraint defined by a low-rank matrix A:
 *
 *     A = sum_{j=1}^k s_j a_j a_j^T
 *
 * Here A is n-by-k. For this to be efficient, you should have k << n.
 */
template <typename MatrixType = arma::mat>
class LowRankConstraint : public virtual LinearConstraint
{
public:
  LowRankConstraint();

  // LinearConstraint interface.
  virtual arma::vec Evaluate(const arma::mat& x);
  virtual arma::vec EvaluateLR(const arma::mat& r);
  virtual arma::mat GradientLR(const arma::mat& r, const arma::vec& y);

  //! Return the vector of A column vectors.
  const std::vector<MatrixType>& A_vector() const { return _a_vector; }

  //! Modify the vector of A column vectors.
  std::vector<MatrixType>& A_vector() { return _a_vector; }

  //! Return the vector of A weights.
  const std::vector<arma::vec>& A_weight() const { return _a_weight; }

  //! Modify the vector of A weights.
  std::vector<arma::vec>& A_weight() { return _a_weight; }

  //! Return the vector of b values.
  const arma::vec& B() const { return _b; }
  
  //! Modify the vector of b values.
  arma::vec& B() { return _b; }

private:
  std::vector<MatrixType> _a_vector;
  std::vector<arma::vec> _a_weight;
  arma::vec _b;
};
  
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "linear_constraint_impl.hpp"

#endif
