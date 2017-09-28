/**
 * @file cyclical_decay.hpp
 * @author Marcus Edel
 *
 * Definition of the warm restart technique (SGDR) described in:
 * "SGDR: Stochastic Gradient Descent with Warm Restarts" by
 * I. Loshchilov et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_SGDR_CYCLICAL_DECAY_HPP
#define MLPACK_CORE_OPTIMIZERS_SGDR_CYCLICAL_DECAY_HPP

namespace mlpack {
namespace optimization {

/**
 * Simulate a new warm-started run/restart once a number of epochs are
 * performed. Importantly, the restarts are not performed from scratch but
 * emulated by increasing the step size while the old step size value of as an
 * initial parameter.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Loshchilov2016,
 *   title   = {Learning representations by back-propagating errors},
 *   author  = {Ilya Loshchilov and Frank Hutter},
 *   title   = {{SGDR:} Stochastic Gradient Descent with Restarts},
 *   journal = {CoRR},
 *   year    = {2016},
 *   url     = {https://arxiv.org/abs/1608.03983}
 * }
 * @endcode
 */
class CyclicalDecay
{
 public:
  /**
   * Construct the CyclicalDecay technique a restart method, where the
   * step size decays after each batch and peridically resets to its initial
   * value.
   *
   * @param epochRestart Initial epoch where decay is applied.
   * @param multFactor Factor to increase the number of epochs before a restart.
   * @param stepSize Initial step size for each restart.
   * @param batchSize Size of each mini-batch.
   * @param numFunctions The number of separable functions (the number of
   *        predictor points).
   */
  CyclicalDecay(const size_t epochRestart,
                const double multFactor,
                const double stepSize) :
      epochRestart(epochRestart),
      multFactor(multFactor),
      constStepSize(stepSize),
      nextRestart(epochRestart),
      batchRestart(0),
      epochBatches(0),
      epoch(0)
  { /* Nothing to do here */ }

  /**
   * This function is called in each iteration after the policy update.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& /* iterate */,
              double& stepSize,
              const arma::mat& /* gradient */)
  {
    // Time to adjust the step size.
    if (epoch >= epochRestart)
    {
      // n_t = n_min^i + 0.5(n_max^i - n_min^i)(1 + cos(T_cur/T_i * pi)).
      stepSize = 0.5 * constStepSize * (1 + cos((batchRestart / epochBatches)
          * M_PI));

      // Keep track of the number of batches since the last restart.
      batchRestart++;
    }

    // Time to restart.
    if (epoch > nextRestart)
    {
      batchRestart = 0;

      // Adjust the period of restarts.
      epochRestart *= multFactor;

      // Update the time for the next restart.
      nextRestart += epochRestart;
    }

    epoch++;
  }

  //! Get the step size.
  double StepSize() const { return constStepSize; }
  //! Modify the step size.
  double& StepSize() { return constStepSize; }

  //! Get the restart fraction.
  double EpochBatches() const { return epochBatches; }
  //! Modify the restart fraction.
  double& EpochBatches() { return epochBatches; }

 private:
  //! Epoch where decay is applied.
  size_t epochRestart;

  //! Parameter to increase the number of epochs before a restart.
  double multFactor;

  //! The step size for each example.
  double constStepSize;

  //! Locally-stored restart time.
  size_t nextRestart;

  //! Locally-stored number of batches since the last restart.
  size_t batchRestart;

  //! Locally-stored restart fraction.
  double epochBatches;

  //! Locally-stored epoch.
  size_t epoch;
};

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_SGDR_CYCLICAL_DECAY_HPP
