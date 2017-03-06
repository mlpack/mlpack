/**
 * @file nag.hpp
 * @author Ryan Curtin
 * @author Krishna Kant Singh
 *
 * Stochastic Gradient Descent(SGD) with Nestrov Accelrated Gradient
 * update(NAG) is an approach 
 * that enjoys accelerated convergence rates compared to vanilla SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_MOMENTUM_SGD_MOMENTUM_SGD_HPP
#define MLPACK_CORE_OPTIMIZERS_MOMENTUM_SGD_MOMENTUM_SGD_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Learning with SGD can sometimes be slow. Hence we use the 
 * Nestrov Accelarated Gradient henceforth NAG.
 * Which is defined as follows.
 * Update the parameters using the present velocity 
 * value 
 * \f[
 * \tilde A_{j} = A_{j} + \alpha*v
 * \f]
 * Compute the gradient at these parameter values and velocity
 * \f[
 * g = \nabla f_j(\tilde A_j) 
 * v = \alpha*v - \epsilon*g
 * \f]
 * Using these values now update \f[A_{j + 1} = A_j + v\f]
 *
 * For more information, please refer to the Section 8.3.3 of the following book
 *
 * @code
 * @book{Goodfellow-et-al-2016,
 *  title={Deep Learning},
 *  author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
 *  publisher={MIT Press},
 *  note={\url{http://www.deeplearningbook.org}},
 *  year={2016}
 * }
 *
 *
 * This class is useful for data-dependent functions whose objective function
 * can be expressed as a sum of objective functions operating on an individual
 * point.  Then, NAG considers the gradient of the objective function operating
 * on an individual point in its update of \f$ A \f$.
 *
 * For NAG to work, a DecomposableFunctionType template parameter is required.
 * This class must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates, const size_t i);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA (see mlpack::nca::NCA), NumFunctions() should return the number
 * of points in the dataset, and Evaluate(coordinates, 0) will evaluate the
 * objective function on the first point in the dataset (presumably, the dataset
 * is held internally in the DecomposableFunctionType).
 *
 * @tparam DecomposableFunctionType Decomposable objective function type to be
 *     minimized.
 */
template<typename DecomposableFunctionType>
class NAG
{
 public:
  /**
   * Construct the NAG optimizer with the given function and parameters.  The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset). Typically the momentum paramter is often
   * initialized with small value like 0.5 and later raised.
   *
   * @param function Function to be optimized (minimized).
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param momentum The momentum hyperparameter
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   */
  NAG(DecomposableFunctionType& function,
      const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const double momentum = 0.5,
      const bool shuffle = true);

  /**
   * Optimize the given function using NAG.  The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate);

  //! Get the instantiated function to be optimized.
  const DecomposableFunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  DecomposableFunctionType& Function() { return function; }

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

  //! Get the momentum parameter.
  double Momentum() const { return momentum; }
  //! Modify the momentum paramete.
  double& Momentum() { return momentum; }

 private:
  //! The instantiated function.
  DecomposableFunctionType& function;

  //! The step size for each example.
  double stepSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! The momentum hyperparameter
  double momentum;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "nag_impl.hpp"

#endif