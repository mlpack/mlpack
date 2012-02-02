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

  double Optimize(arma::mat& coordinates);
//                AugLagrangian<LRSDP> auglag);

  double Evaluate(const arma::mat& coordinates) const;

  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const;

  double EvaluateConstraint(const size_t index,
                            const arma::mat& coordinates) const;

  void GradientConstraint(const size_t index,
                          const arma::mat& coordinates,
                          arma::mat& gradient) const;

  size_t NumConstraints() const { return b.n_elem; }

  const arma::mat& GetInitialPoint();

  const arma::mat& C() const { return c; }
  arma::mat& C() { return c; }

  const std::vector<arma::mat> A() const { return a; }
  std::vector<arma::mat>& A() { return a; }

  const arma::vec& B() const { return b; }
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
