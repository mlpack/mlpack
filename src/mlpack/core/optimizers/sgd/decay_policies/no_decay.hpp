/**
 * @file no_decay.hpp
 * @author Marcus Edel
 *
 * Definition of the policy type for the decay class.
 *
 * You should define your own decay update that looks like NoDecay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_MINIBATCH_SGD_DECAY_POLICIES_NO_DECAY_HPP
#define MLPACK_CORE_OPTIMIZERS_MINIBATCH_SGD_DECAY_POLICIES_NO_DECAY_HPP

namespace mlpack {
namespace optimization {

/**
 * Definition of the NoDecay class. Use this as a template for your own.
 */
class NoDecay
{
 public:
  /**
   * This constructor is called before the first iteration.
   */
  NoDecay() { }

  /**
   * This function is called in each iteration after the policy update.
   *
  * @param iterate Parameters that minimize the function.
  * @param stepSize Step size to be used for the given iteration.
  * @param gradient The gradient matrix.
  */
  void Update(arma::mat& /* iterate */,
              double& /* stepSize */,
              const arma::mat& /* gradient */)
  {
    // Nothing to do here.
  }

  /**
   * This function is called in each iteration after the SVRG update step.
   *
   * @param iterate Parameters that minimize the function.
   * @param iterate0 The last function parameters at time t - 1.
   * @param gradient The current gradient matrix at time t.
   * @param fullGradient The computed full gradient.
   * @param stepSize Step size to be used for the given iteration.
   */
  void Update(const arma::mat& /* iterate */,
              const arma::mat& /*iterate0 */,
              const arma::mat& /* gradient */,
              const arma::mat& /* fullGradient */,
              const size_t /* numBatches */,
              double& /* stepSize */)
  {
    // Nothing to do here.
  }
};

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_MINIBATCH_SGD_DECAY_POLICIES_NO_DECAY_HPP
