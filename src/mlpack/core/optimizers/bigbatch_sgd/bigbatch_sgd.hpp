/**
 * @file bigbatch_sgd.hpp
 * @author Marcus Edel
 *
 * Definition of Big-batch Stochastic Gradient Descent (BBSGD) as described in:
 * "Big Batch SGD: Automated Inference using Adaptive Batch Sizes"
 * by Soham De et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BIGBATCH_SGD_HPP
#define MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BIGBATCH_SGD_HPP

#include <mlpack/prereqs.hpp>

#include "adaptive_stepsize.hpp"
#include "backtracking_line_search.hpp"

namespace mlpack {
namespace optimization {

/**
 * Big-batch Stochastic Gradient Descent is a technique for minimizing a
 * function which can be expressed as a sum of other functions.  That is,
 * suppose we have
 *
 * \f[
 * f(A) = \sum_{i = 0}^{n} f_i(A)
 * \f]
 *
 * and our task is to minimize \f$ A \f$.  Big-batch SGD iterates over batches
 * of functions \f$ \{ f_{i0}(A), f_{i1}(A), \ldots, f_{i(m - 1)}(A) \f$ for
 * some batch size \f$ m \f$, producing the following update scheme:
 *
 * \f[
 * A_{j + 1} = A_j + \alpha \left(\sum_{k = 0}^{m - 1} \nabla f_{ik}(A) \right)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  Each
 * big-batch is passed through either sequentially or randomly.  The algorithm
 * continues until \f$ j \f$ reaches the maximum number of iterations---or when
 * a full sequence of updates through each of the big-batches produces an
 * improvement within a certain tolerance \f$ \epsilon \f$.
 *
 * The parameter \f$ \epsilon \f$ is specified by the tolerance parameter tot he
 * constructor, as is the maximum number of iterations specified by the
 * maxIterations parameter.
 *
 * This class is useful for data-dependent functions whose objective function
 * can be expressed as a sum of objective functions operating on an individual
 * point.  Then, big-batch SGD considers the gradient of the objective function
 * operation on an individual big-batch of points in its update of \f$ A \f$.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{De2017,
 *   title   = {Big Batch {SGD:} Automated Inference using Adaptive Batch
 *              Sizes},
 *   author  = {Soham De and Abhay Kumar Yadav and David W. Jacobs and
                Tom Goldstein},
 *   journal = {CoRR},
 *   year    = {2017},
 *   url     = {http://arxiv.org/abs/1610.05792},
 * }
 * @endcode
 *
 * For big-batch SGD to work, a DecomposableFunctionType template parameter is
 * required.
 * This class must implement the following function:
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
 * @tparam UpdatePolicyType Update policy used during the iterative update
 *     process. By default the AdaptiveStepsize update policy is used.
 */
template<typename UpdatePolicyType = AdaptiveStepsize>
class BigBatchSGD
{
 public:
  /**
   * Construct the BigBatchSGD optimizer with the given function and
   * parameters.  The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored for the task
   * at hand.  The maximum number of iterations refers to the maximum number of
   * batches that are processed.
   *
   * @param batchSize Initial batch size.
   * @param stepSize Step size for each iteration.
   * @param batchDelta Factor for the batch update step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the batch order is shuffled; otherwise, each
   *        batch is visited in linear order.
   */
  BigBatchSGD(const size_t batchSize = 1000,
              const double stepSize = 0.01,
              const double batchDelta = 0.1,
              const size_t maxIterations = 100000,
              const double tolerance = 1e-5,
              const bool shuffle = true);
  /**
   * Optimize the given function using big-batch SGD.  The given starting point
   * will be modified to store the finishing point of the algorithm, and the
   * final objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function,
                  arma::mat& iterate);

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the batch delta.
  double BatchDelta() const { return batchDelta; }
  //! Modify the batch delta.
  double& BatchDelta() { return batchDelta; }

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

  //! Get the update policy.
  UpdatePolicyType UpdatePolicy() const { return updatePolicy; }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy() { return updatePolicy; }

 private:
  //! The size of the current batch.
  size_t batchSize;

  //! The step size for each example.
  double stepSize;

  //! The factor for the batch update step.
  double batchDelta;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;

  //! The update policy used to update the parameters in each iteration.
  UpdatePolicyType updatePolicy;
};

using BBS_Armijo = BigBatchSGD<BacktrackingLineSearch>;
using BBS_BB = BigBatchSGD<AdaptiveStepsize>;

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "bigbatch_sgd_impl.hpp"

#endif
