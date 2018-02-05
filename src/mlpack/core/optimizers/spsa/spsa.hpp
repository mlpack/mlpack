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
  SPSA(const float& alpha = 0.602,
       const float& gamma = 0.101,
       const float& a = 1e-6,
       const float& c = 0.01,
       const size_t& maxIterations = 100000)
  {
    spsaUpdate = new SPSAUpdate(alpha, gamma, a, c,
                      maxIterations);
  }

  ~SPSA()
  {
    delete spsaUpdate;
  }

  template<typename DecomposableFunctionType>
  void Optimize(DecomposableFunctionType& function, arma::mat& iterate)
  {
    spsaUpdate->Update(iterate, function);
  }

  float Alpha() const { return spsaUpdate->Alpha(); }

  float& Alpha() { return spsaUpdate->Alpha(); }

  float Gamma() const { return spsaUpdate->Gamma(); }

  float& Gamma() { return spsaUpdate->Gamma(); }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return spsaUpdate->MaxIterations(); }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return spsaUpdate->MaxIterations(); }

  float Gradient_scaling_parameter(const int& choice) const
  {
    return spsaUpdate->Gradient_scaling_parameter(choice);
  }

  float& Gradient_scaling_parameter(const int& choice)
  {
    return spsaUpdate->Gradient_scaling_parameter(choice);
  }

  float Noise_variance_parameter() const {
           return spsaUpdate->Noise_variance_parameter(); }

  float& Noise_variance_parameter() {
           return spsaUpdate->Noise_variance_parameter(); }
 private:
  //! The SPSA Descent object pointer.
  SPSAUpdate *spsaUpdate;
};

} // namespace optimization
} // namespace mlpack

#endif
