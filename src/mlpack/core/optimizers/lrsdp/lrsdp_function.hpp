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
  size_t NumSparseConstraints() const { return sparseB.n_elem; }

  //! Get the number of dense constraints in the LRSDP.
  size_t NumDenseConstraints() const { return denseB.n_elem; }

  //! Get the total number of constraints in the LRSDP.
  size_t NumConstraints() const {
    return NumSparseConstraints() + NumDenseConstraints();
  }

  //! Get the initial point of the LRSDP.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  size_t n() const { return initialPoint.n_rows; }

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

  bool hasSparseObjective() const { return hasModifiedSparseObjective; }

  bool hasDenseObjective() const { return hasModifiedDenseObjective; }

  //! Return string representation of object.
  std::string ToString() const;

 private:
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
