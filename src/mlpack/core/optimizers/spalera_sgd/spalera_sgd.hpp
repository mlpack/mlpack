/**
 * @file spalera_sgd.hpp
 * @author Marcus Edel
 *
 * SPALeRA Stochastic Gradient Descent (SPALeRASGD).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SPALERA_SGD_SPALERA_SGD_HPP
#define MLPACK_CORE_OPTIMIZERS_SPALERA_SGD_SPALERA_SGD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/spalera_sgd/spalera_stepsize.hpp>
#include <mlpack/core/optimizers/sgd/decay_policies/no_decay.hpp>

namespace mlpack {
namespace optimization {

/**
 * SPALeRA Stochastic Gradient Descent is a technique for minimizing a
 * function which can be expressed as a sum of other functions.  That is,
 * suppose we have
 *
 * \f[
 * f(A) = \sum_{i = 0}^{n} f_i(A)
 * \f]
 *
 * and our task is to minimize \f$ A \f$.  SPALeRA SGD iterates over batches
 * of functions \f$ \{ f_{i0}(A), f_{i1}(A), \ldots, f_{i(m - 1)}(A) \f$ for
 * some batch size \f$ m \f$, producing the following update scheme:
 *
 * \f[
 * A_{j + 1} = A_j + \alpha \left(\sum_{k = 0}^{m - 1} \nabla f_{ik}(A) \right)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  Each
 * batch is passed through either sequentially or randomly.  The algorithm
 * continues until \f$ j \f$ reaches the maximum number of iterations---or when
 * a full sequence of updates through each of the batches produces an
 * improvement within a certain tolerance \f$ \epsilon \f$.
 *
 * The parameter \f$ \epsilon \f$ is specified by the tolerance parameter tot he
 * constructor, as is the maximum number of iterations specified by the
 * maxIterations parameter.
 *
 * This class is useful for data-dependent functions whose objective function
 * can be expressed as a sum of objective functions operating on an individual
 * point.  Then, SPALeRA SGD considers the gradient of the objective function
 * operation on an individual batches in its update of \f$ A \f$.
 *
 * For more information, please refer to:
 *
 * @code
 * @misc{Schoenauer2017,
 *   title   = {Stochastic Gradient Descent:
 *              Going As Fast As Possible But Not Faster},
 *   author  = {Schoenauer-Sebag, Alice; Schoenauer, Marc; Sebag, Michele},
 *   journal = {CoRR},
 *   year    = {2017},
 *   url     = {https://arxiv.org/abs/1709.01427},
 * }
 * @endcode
 *
 * For SPALeRA SGD to work, the lass must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates, const size_t i);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient);
 *
 * NumFunctions() should return the number of functions, and in the other two
 * functions, the parameter i refers to which individual function (or gradient)
 * is being evaluated.  So, for the case of a data-dependent function, such as
 * NCA (see mlpack::nca::NCA), NumFunctions() should return the number of points
 * in the dataset, and Evaluate(coordinates, 0) will evaluate the objective
 * function on the first point in the dataset (presumably, the dataset is held
 * internally in the DecomposableFunctionType).
 *
 * @tparam DecayPolicyType Decay policy used during the iterative update
 *     process to adjust the step size. By default the step size isn't going to
 *     be adjusted.
 */
template<typename DecayPolicyType = NoDecay>
class SPALeRASGD
{
 public:
  /**
   * Construct the SPALeRASGD optimizer with the given function and parameters.
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Batch size to use for each step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param lambda Page-Hinkley update parameter.
   * @param alpha Memory parameter of the Agnostic Learning Rate adaptation.
   * @param epsilon Numerical stability parameter.
   * @param adaptRate Agnostic learning rate update rate.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *    function is visited in linear order.
   * @param decayPolicy Instantiated decay policy used to adjust the step size.
   * @param resetPolicy Flag that determines whether update policy parameters
   *    are reset before every Optimize call.
   */
  SPALeRASGD(const double stepSize = 0.01,
             const size_t batchSize = 32,
             const size_t maxIterations = 100000,
             const double tolerance = 1e-5,
             const double lambda = 0.01,
             const double alpha = 0.001,
             const double epsilon = 1e-6,
             const double adaptRate = 3.10e-8,
             const bool shuffle = true,
             const DecayPolicyType& decayPolicy = DecayPolicyType(),
             const bool resetPolicy = true);

  /**
   * Optimize the given function using SPALeRA SGD.  The given starting point
   * will be modified to store the finishing point of the algorithm, and the
   * final objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

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

  //! Get the tolerance for termination.
  double Alpha() const { return updatePolicy.Alpha(); }
  //! Modify the tolerance for termination.
  double& Alpha() { return updatePolicy.Alpha(); }

  //! Get the agnostic learning rate update rate.
  double AdaptRate() const { return updatePolicy.AdaptRate(); }
  //! Modify the agnostic learning rate update rate.
  double& AdaptRate() { return updatePolicy.AdaptRate(); }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

  //! Get whether or not the update policy parameters
  //! are reset before Optimize call.
  bool ResetPolicy() const { return resetPolicy; }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return resetPolicy; }

  //! Get the update policy.
  SPALeRAStepsize UpdatePolicy() const { return updatePolicy; }
  //! Modify the update policy.
  SPALeRAStepsize& UpdatePolicy() { return updatePolicy; }

  //! Get the decay policy.
  DecayPolicyType DecayPolicy() const { return decayPolicy; }
  //! Modify the decay policy.
  DecayPolicyType& DecayPolicy() { return decayPolicy; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The batch size for processing.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Page-Hinkley update parameter.
  double lambda;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;

  //! The update policy used to update the parameters in each iteration.
  SPALeRAStepsize updatePolicy;

  //! The decay policy used to update the parameters in each iteration.
  DecayPolicyType decayPolicy;

  //! Flag that determines whether update policy parameters
  //! are reset before every Optimize call.
  bool resetPolicy;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "spalera_sgd_impl.hpp"

#endif
