/**
 * @file trust_region_newton.hpp
 * @author Marcus Edel
 *
 * Definition of the Trust Region Newton Method as described in
 * "Trust Region Newton Method for Large-Scale Logistic Regression"
 * by Chih-Jen Lin et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_TRUST_REGION_NEWTON_TRUST_REGION_NEWTON_HPP
#define MLPACK_CORE_OPTIMIZERS_TRUST_REGION_NEWTON_TRUST_REGION_NEWTON_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * The Trust Region Newton optimizer, which uses approximate Newton steps in
 * the beginning to maximize the log-likelihood. The parameters for the
 * algorithm (maximum number of iterations, gradient norm and so forth)
 * are all configurable via either the constructor or standalone modifier
 * functions.
 *
 * For more information, please refer to:
 *
 * @code
 * @inproceedings{Lin2007,
 *   author = {Lin, Chih-Jen and Weng, Ruby C. and Keerthi, S. Sathiya},
 *   title = {Trust Region Newton Methods for Large-scale Logistic Regression},
 *   booktitle = {Proceedings of the 24th International Conference on Machine
 *       Learning},
 *   series = {ICML '07},
 *   year = {2007},
 *   pages = {561--568},
 *   publisher = {ACM}
 * }
 * @endcode
 *
 * A function which can be optimized by this class must implement
 * the following methods:
 *
 *  - a default constructor
 *  - double Evaluate(const arma::mat& coordinates);
 *  - void Gradient(const arma::mat& coordinates,
 *                  arma::mat& gradient,
 *                  arma::mat& derivative);
 *  - void Hessian(const arma::mat& coordinates,
 *                 const arma::mat& gradient,
 *                 const arma::mat& derivative,
 *                 arma::mat& hessian);
 */
template<typename FunctionType>
class TrustRegionNewton
{
 public:
  /**
   * Construct the Trust Region Newton optimizer with the given function and
   * parameters.  The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task
   * at hand.
   *
   * @param function Function to be optimized (minimized).
   * @param minGradientNorm Minimum gradient norm required to continue the
   *        optimization.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param maxConjugateIterations Maximum number of iterations allowed for the
   *        conjugate procedure  (0 means no limit).
   * @param eta0 Positive constant to update the update rule select: such that
   *        eta0 < eta1 < eta2 < 1.
   * @param eta1 Positive constant to update the update rule select: such that
   *        eta0 < eta1 < eta2 < 1.
   * @param eta2 Positive constant to update the update rule select: such that
   *        eta0 < eta1 < eta2 < 1.
   * @param sigma1 Positive constant to update delta k select: such that
   *        sigma1 < sigma2 < 1 < sigma3 (Default 0.25).
   * @param sigma2 Positive constant to update delta k select: such that
   *        sigma1 < sigma2 < 1 < sigma3 (Default 0.5).
   * @param sigma3 Positive constant to update delta k select: such that
   *        sigma1 < sigma2 < 1 < sigma3 (Default 4).
   */
  TrustRegionNewton(FunctionType& function,
                    const double minGradientNorm = 1e-6,
                    const size_t maxIterations = 20,
                    const size_t maxConjugateIterations = 0,
                    const double eta0 = 1e-4,
                    const double eta1 = 0.25,
                    const double eta2 = 0.75,
                    const double sigma1 = 0.25,
                    const double sigma2 = 0.5,
                    const double sigma3 = 4);

  /**
   * Optimize the given function using the Trust Region Newton method. The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(FunctionType& function, arma::mat& iterate);

  /**
   * Optimize the given function using the Trust Region Newton method. The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate)
  {
    return Optimize(this->function, iterate);
  }

  //! Get the instantiated function to be optimized.
  const FunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  FunctionType& Function() { return function; }

  //! Get the min gradient norm.
  double MinGradientNorm() const { return minGradientNorm; }
  //! Modify the min gradient norm.
  double& MinGradientNorm() { return minGradientNorm; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

 private:
  /*
   * Conjugate gradient procedure for approximately solving the trust region
   * sub-problem.
   *
   * @param function Function to optimize.
   * @param delta Scaling factor to use for the trust region sub-problem.
   * @param iterate The initial point to begin with solving the trust region
   *                sub-problem.
   */
  void ConjugateGradient(FunctionType& function,
                         const double delta,
                         const arma::mat& iterate);

  //! The instantiated function.
  FunctionType& function;

  //! Minimum gradient norm required to continue the optimization.
  double minGradientNorm;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The maximum number of allowed iterations for the conjugate procedure.
  size_t maxConjugateIterations;

  //! Update rule rate.
  double eta0;

  //! Update rule rate.
  double eta1;

  //! Update rule rate.
  double eta2;

  //! Scaling update factor.
  double sigma1;

  //! Scaling update factor.
  double sigma2;

  //! Scaling update factor.
  double sigma3;

  //! Scores the the update matrix s.
  arma::mat s;

  //! Scores the the update matrix r.
  arma::mat r;

  //! Internally stored derivative.
  arma::mat derivative;

  //! Internally stored gradient.
  arma::mat gradient;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "trust_region_newton_impl.hpp"

#endif
