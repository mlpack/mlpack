/**
 * @file spsa_update.hpp
 * @author N Rajiv Vaidyanathan
 *
 * SPSA update for Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SPSA_SPSA_HPP
#define MLPACK_CORE_OPTIMIZERS_SPSA_SPSA_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include "spsa_update.hpp"

namespace mlpack {
namespace optimization {

/**
 * Implementation of the SPSA update policy. SPSA update policy improves
 * the rate of optimization by performing simultaneously on all parameters.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Spall1998,
 *   author  = {Spall, J. C.},
 *   title   = {An Overview of the Simultaneous Perturbation Method for
 *              Efficient Optimization},
 *   journal = {Johns Hopkins APL Technical Digest},
 *   volume  = {19},
 *   number  = {4},
 *   pages   = {482--492},
 *   year    = {1998}
 * }
 * @endcode
 *
 */
class SPSA
{
 public:
  SPSA(const double stepSize = 0.01,
       const size_t batchSize = 32,
       const float& alpha = 0.602,
       const float& gamma = 0.101,
       const float& a = 1e-6,
       const float& c = 0.01,
       const size_t& maxIterations = 100000,
       const double& tolerance = 1e-5,
       const bool& shuffle = true);

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

  float Alpha() const { return optimizer.UpdatePolicy().Alpha(); }

  float& Alpha() { return optimizer.UpdatePolicy().Alpha(); }

  float Gamma() const { return optimizer.UpdatePolicy().Gamma(); }

  float& Gamma() { return optimizer.UpdatePolicy().Gamma(); }

  float Gradient_scaling_parameter(const int choice) const
  {
    return optimizer.UpdatePolicy()
          .Gradient_scaling_parameter(choice);
  }

  float& Gradient_scaling_parameter(const int& choice)
  {
    return optimizer.UpdatePolicy()
          .Gradient_scaling_parameter(choice);
  }

  float Noise_variance_parameter() const { return optimizer.UpdatePolicy()
                                  .Noise_variance_parameter(); }

  float& Noise_variance_parameter() { return optimizer.UpdatePolicy()
                                  .Noise_variance_parameter(); }

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
  //! The Stochastic Gradient Descent object with AdaGrad policy.
  SGD<SPSAUpdate> optimizer;
};

} // namespace optimization
} // namespace mlpack

#endif
