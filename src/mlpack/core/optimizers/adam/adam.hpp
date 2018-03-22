/**
 * @file adam.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Marcus Edel
 * @author Vivek Pal
 * @author Sourabh Varshney
 * @author Haritha Nair
 *
 * Adam, AdaMax, AMSGrad, Nadam and Nadamax optimizers. Adam is an an algorithm
 * for first-order gradient-based optimization of stochastic objective
 * functions, based on adaptive estimates of lower-order moments. AdaMax is
 * simply a variant of Adam based on the infinity norm. AMSGrad is another
 * variant of Adam with guaranteed convergence. Nadam is another variant of 
 * Adam based on NAG. NadaMax is a variant for Nadam based on Infinity form.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_HPP
#define MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include "adam_update.hpp"
#include "adamax_update.hpp"
#include "amsgrad_update.hpp"
#include "nadam_update.hpp"
#include "nadamax_update.hpp"
#include "optimisticadam_update.hpp"

namespace mlpack {
namespace optimization {

/**
 * Adam is an optimizer that computes individual adaptive learning rates for
 * different parameters from estimates of first and second moments of the
 * gradients. AdaMax is a variant of Adam based on the infinity norm as given
 * in the section 7 of the following paper. Nadam is an optimizer that
 * combines the Adam and NAG. NadaMax is an variant of Nadam based on Infinity
 * form.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Kingma2014,
 *   author  = {Diederik P. Kingma and Jimmy Ba},
 *   title   = {Adam: {A} Method for Stochastic Optimization},
 *   journal = {CoRR},
 *   year    = {2014},
 *   url     = {http://arxiv.org/abs/1412.6980}
 * }
 * @article{
 *   title   = {On the convergence of Adam and beyond},
 *   url     = {https://openreview.net/pdf?id=ryQu7f-RZ}
 *   year    = {2018}
 * }
 * @endcode
 *
 * For Adam, AdaMax, AMSGrad, Nadam and NadaMax to work, a DecomposableFunctionType
 * template parameter is required. This class must implement the following
 * function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates,
 *                   const size_t i,
 *                   const size_t batchSize);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient,
 *                 const size_t batchSize);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA (see mlpack::nca::NCA), NumFunctions() should return the number
 * of points in the dataset, and Evaluate(coordinates, 0) will evaluate the
 * objective function on the first point in the dataset (presumably, the dataset
 * is held internally in the DecomposableFunctionType).
 *
 * @tparam UpdateRule Adam optimizer update rule to be used.
 */
template<typename UpdateRule = AdamUpdate>
class AdamType
{
 public:
  /**
   * Construct the Adam optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
            estimates.
   * @param eps Value used to initialise the mean squared gradient parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   */
  AdamType(const double stepSize = 0.001,
           const size_t batchSize = 32,
           const double beta1 = 0.9,
           const double beta2 = 0.999,
           const double eps = 1e-8,
           const size_t maxIterations = 100000,
           const double tolerance = 1e-5,
           const bool shuffle = true);

  /**
   * Optimize the given function using Adam. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to optimize.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate)
  {
    return optimizer.Optimize(function, iterate);
  }

  //! Get the step size.
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the smoothing parameter.
  double Beta1() const { return optimizer.UpdatePolicy().Beta1(); }
  //! Modify the smoothing parameter.
  double& Beta1() { return optimizer.UpdatePolicy().Beta1(); }

  //! Get the second moment coefficient.
  double Beta2() const { return optimizer.UpdatePolicy().Beta2(); }
  //! Modify the second moment coefficient.
  double& Beta2() { return optimizer.UpdatePolicy().Beta2(); }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return optimizer.UpdatePolicy().Epsilon(); }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return optimizer.UpdatePolicy().Epsilon(); }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return optimizer.MaxIterations(); }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return optimizer.MaxIterations(); }

  //! Get the tolerance for termination.
  double Tolerance() const { return optimizer.Tolerance(); }
  //! Modify the tolerance for termination.
  double& Tolerance() { return optimizer.Tolerance(); }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return optimizer.Shuffle(); }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return optimizer.Shuffle(); }

 private:
  //! The Stochastic Gradient Descent object with Adam policy.
  SGD<UpdateRule> optimizer;
};

using Adam = AdamType<AdamUpdate>;

using AdaMax = AdamType<AdaMaxUpdate>;

using AMSGrad = AdamType<AMSGradUpdate>;

using Nadam = AdamType<NadamUpdate>;

using NadaMax = AdamType<NadaMaxUpdate>;

using OptimisticAdam = AdamType<OptimisticAdamUpdate>;

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "adam_impl.hpp"

#endif
