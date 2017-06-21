/**
 * @file cdk.hpp
 *
 *
 * Implementation of stochastic gradient descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CDK_CDK_HPP
#define MLPACK_CORE_OPTIMIZERS_CDK_CDK_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization /** Artificial Neural Network. */ {
/*
 * The cdk algorithm for training RBM.
 * rbmtype is function that implements
 * the following functions
 * n_functions: number of data points
 * gradients: gradient function for calculating the positive and negative gradient at the given point
 * gibbs: gibbs sampler for obtaining the samples for given variable after k-steps
 *
 * @tparam: RBMType: Type of RBM being used
 */
template<typename RBMType>
class CDK
{
 public:
  /**
   * The default constructor for the CD-k aglorithm
   *
   * @tparam: RBMType: RBM for which we want to train the algorithm
   * @param: epoch: Number of training steps
   * @param: k: chain length of gibbs sampler
   * @param: persistent: PCD-k or CD-k
   */
  CDK(RBMType& rbm,
      const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const size_t batchSize = 20,
      const bool shuffle = true);

  /**
   * Optimize the given function using cd-k.  The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  void Optimize(arma::mat& iterate);

  //! Get the instantiated function to be optimized.
  const RBMType& RBM() const { return rbm; }
  //! Modify the instantiated function.
  RBMType& RBM() { return rbm; }

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

 private:
  //! The instantiated function.
  RBMType& rbm;

  //! The step size for each example.
  const double stepSize;

  //! The maximum number of allowed iterations.
  const size_t maxIterations;

  //! Batch size.
  const size_t batchSize;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  const bool shuffle;

  // negative_sample: The negative sample
  arma::mat negative_sample;
};
} // namespace optimization
} // namespace mlpack
#include "cdk_impl.hpp"
#endif
