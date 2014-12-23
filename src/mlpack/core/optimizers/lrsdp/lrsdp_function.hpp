/**
 * @file lrsdp_function.hpp
 * @author Ryan Curtin
 * @author Abhishek Laddha
 *
 * A class that represents the objective function which LRSDP optimizes.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_FUNCTION_HPP
#define __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_FUNCTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

namespace mlpack {
namespace optimization {

/**
 * The objective function that LRSDP is trying to optimize.
 */
class LRSDPFunction
{
 public:
  /**
   * Construct the LRSDPFunction with the given initial point and number of
   * constraints. Note n_cols of the initialPoint specifies the rank.
   *
   * Set the A_x, B_x, and C_x  matrices for each constraint using the A_x(),
   * B_x(), and C_x() functions, for x in {sparse, dense}.
   */
  LRSDPFunction(const size_t numSparseConstraints,
                const size_t numDenseConstraints,
                const arma::mat& initialPoint);

  /**
   * Evaluate the objective function of the LRSDP (no constraints) at the given
   * coordinates.
   */
  double Evaluate(const arma::mat& coordinates) const;

  /**
   * Evaluate the gradient of the LRSDP (no constraints) at the given
   * coordinates.
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

  //! Get the number of sparse constraints in the LRSDP.
  size_t NumSparseConstraints() const { return b_sparse.n_elem; }

  //! Get the number of dense constraints in the LRSDP.
  size_t NumDenseConstraints() const { return b_dense.n_elem; }

  //! Get the total number of constraints in the LRSDP.
  size_t NumConstraints() const {
    return NumSparseConstraints() + NumDenseConstraints();
  }

  //! Get the initial point of the LRSDP.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  size_t n() const { return initialPoint.n_rows; }

  //! Return the sparse objective function matrix (C_sparse).
  const arma::sp_mat& C_sparse() const { return c_sparse; }

  //! Modify the sparse objective function matrix (C_sparse).
  arma::sp_mat& C_sparse() {
    hasModifiedSparseObjective = true;
    return c_sparse;
  }

  //! Return the dense objective function matrix (C_dense).
  const arma::mat& C_dense() const { return c_dense; }

  //! Modify the dense objective function matrix (C_dense).
  arma::mat& C_dense() {
    hasModifiedDenseObjective = true;
    return c_dense;
  }

  //! Return the vector of sparse A matrices (which correspond to the sparse
  // constraints).
  const std::vector<arma::sp_mat>& A_sparse() const { return a_sparse; }

  //! Modify the veector of sparse A matrices (which correspond to the sparse
  // constraints).
  std::vector<arma::sp_mat>& A_sparse() { return a_sparse; }

  //! Return the vector of dense A matrices (which correspond to the dense
  // constraints).
  const std::vector<arma::mat>& A_dense() const { return a_dense; }

  //! Modify the veector of dense A matrices (which correspond to the dense
  // constraints).
  std::vector<arma::mat>& A_dense() { return a_dense; }

  //! Return the vector of sparse B values.
  const arma::vec& B_sparse() const { return b_sparse; }
  //! Modify the vector of sparse B values.
  arma::vec& B_sparse() { return b_sparse; }

  //! Return the vector of dense B values.
  const arma::vec& B_dense() const { return b_dense; }
  //! Modify the vector of dense B values.
  arma::vec& B_dense() { return b_dense; }

  bool hasSparseObjective() const { return hasModifiedSparseObjective; }

  bool hasDenseObjective() const { return hasModifiedDenseObjective; }

  //! Return string representation of object.
  std::string ToString() const;

 private:
  //! Sparse objective function matrix c.
  arma::sp_mat c_sparse;

  //! Dense objective function matrix c.
  arma::mat c_dense;

  //! If false, c_sparse is zero
  bool hasModifiedSparseObjective;

  //! If false, c_dense is zero
  bool hasModifiedDenseObjective;

  //! A_i for each sparse constraint.
  std::vector<arma::sp_mat> a_sparse;
  //! b_i for each sparse constraint.
  arma::vec b_sparse;

  //! A_i for each dense constraint.
  std::vector<arma::mat> a_dense;
  //! b_i for each dense constraint.
  arma::vec b_dense;

  //! Initial point.
  arma::mat initialPoint;
};

// Declare specializations in lrsdp_function.cpp.
template<>
double AugLagrangianFunction<LRSDPFunction>::Evaluate(
    const arma::mat& coordinates) const;

template<>
void AugLagrangianFunction<LRSDPFunction>::Gradient(
    const arma::mat& coordinates,
    arma::mat& gradient) const;

};
};

#endif // __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_FUNCTION_HPP
