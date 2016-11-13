/**
 * @file lrsdp_function.hpp
 * @author Ryan Curtin
 * @author Abhishek Laddha
 *
 * A class that represents the objective function which LRSDP optimizes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_FUNCTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
#include <mlpack/core/optimizers/sdp/sdp.hpp>

namespace mlpack {
namespace optimization {

/**
 * The objective function that LRSDP is trying to optimize.
 */
template <typename SDPType>
class LRSDPFunction
{
 public:

  /**
   * Construct the LRSDPFunction from the given SDP.
   *
   * @param sdp
   * @param initialPoint
   */
  LRSDPFunction(const SDPType& sdp,
                const arma::mat& initialPoint);

  /**
   * Construct the LRSDPFunction with the given initial point and number of
   * constraints. Note n_cols of the initialPoint specifies the rank.
   *
   * Set the A_x, B_x, and C_x  matrices for each constraint using the A_x(),
   * B_x(), and C_x() functions, for x in {sparse, dense}.
   *
   * @param numSparseConstraints
   * @param numDenseConstraints
   * @param initialPoint
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

  //! Get the total number of constraints in the LRSDP.
  size_t NumConstraints() const { return sdp.NumConstraints(); }

  //! Get the initial point of the LRSDP.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  //! Return the SDP object representing the problem.
  const SDPType& SDP() const { return sdp; }

  //! Modify the SDP object representing the problem.
  SDPType& SDP() { return sdp; }

 private:

  //! SDP object representing the problem
  SDPType sdp;

  //! Initial point.
  arma::mat initialPoint;
};

// Declare specializations in lrsdp_function.cpp.
template <>
inline double AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Evaluate(
    const arma::mat& coordinates) const;

template <>
inline double AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Evaluate(
    const arma::mat& coordinates) const;

template <>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Gradient(
    const arma::mat& coordinates,
    arma::mat& gradient) const;

template <>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Gradient(
    const arma::mat& coordinates,
    arma::mat& gradient) const;

} // namespace optimization
} // namespace mlpack

// Include implementation
#include "lrsdp_function_impl.hpp"

#endif // MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_FUNCTION_HPP
