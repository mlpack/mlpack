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
   *
   * @param node Node which this corresponds to.
   */
  NoDecay() { }

  /**
   * This function is called in each iteration after the policy update.
   *
   * @param stepSize The stepSize to be adjusted.
   * @param epoch The current epoch.
   * @param batch The current batch.
   * @param iterate Function parameters.
   */
  void Update(double& /* stepSize */,
              const size_t /* epoch */,
              const size_t /* batch */,
              const arma::mat& /* iterate */)
  {
   // Nothing to do here.
  }
};

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_MINIBATCH_SGD_DECAY_POLICIES_NO_DECAY_HPP