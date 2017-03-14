/**
 * @file momentum_sgd.hpp
 * @author Ryan Curtin
 * @author Arun Reddy
 *
 * Stochastic Gradient Descent(SGD) with Momentum update(MomentumSGD) is an approach
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
 * Learning with SGD can sometimes be slow. Using momentum update for parameter
 * learning can accelerate the rate of convergence, specifically in the cases
 * where the surface curves much more steeply(a steep hilly terrain with high curvature)
 * . The momentum algorithm introduces a new velocity vector \f$ v \f$ with the
 * same dimension as the paramater \f$ A \f$. Also it introduces a new hyperparameter
 * momentum \f$ mu \in (0,1] \f$. Common values of \f$ mu \f$ include 0.5, 0.9 and 0.99.
 * Typically it begins with a small value and later raised.
 *
 * For more information, please refer to the Section 8.3.2 of the following book
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
 * SGD is a technique for minimizing a function which can be expressed as a
 * sum of other functions.  That is, suppose we have
 *
 * \f[
 * f(A) = \sum_{i = 0}^{n} f_i(A)
 * \f]
 *
 * and our task is to minimize \f$ A \f$.  Unlike SGD which directly uses the
 * gradient of the function \f$ A \f$ , SGD with momentum update iterates over
 * each function \f$ f_i(A) \f$, producing the following update scheme:
 *
 * \f[
 * v = mu*v - \alpha \nabla f_i(A)
 * A_{j + 1} = A_j + v
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  \f$ i \f$
 * is chosen according to \f$ j \f$ (the iteration number).  The MomentumSGD class
 * supports either scanning through each of the \f$ n \f$ functions \f$ f_i(A)
 * \f$ linearly, or in a random sequence.  The algorithm continues until \f$ j
 * \f$ reaches the maximum number of iterations---or when a full sequence of
 * updates through each of the \f$ n \f$ functions \f$ f_i(A) \f$ produces an
 * improvement within a certain tolerance \f$ \epsilon \f$.  That is,
 *
 * \f[
 * | f(A_{j + n}) - f(A_j) | < \epsilon.
 * \f]
 *
 * The parameter \f$\epsilon\f$ is specified by the tolerance parameter to the
 * constructor; \f$n\f$ is specified by the maxIterations parameter.
 *
 * This class is useful for data-dependent functions whose objective function
 * can be expressed as a sum of objective functions operating on an individual
 * point.  Then, MomentumSGD considers the gradient of the objective function operating
 * on an individual point in its update of \f$ A \f$.
 *
 * For MomentumSGD to work, a DecomposableFunctionType template parameter is required.
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
class MomentumSGD
{
 public:
  /**
   * Construct the MomentumSGD optimizer with the given function and parameters.  The
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
  MomentumSGD(DecomposableFunctionType& function,
      const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const double momentum = 0.5,
      const bool shuffle = true);

  /**
   * Optimize the given function using stochastic gradient descent.  The given
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
#include "momentum_sgd_impl.hpp"

#endif
