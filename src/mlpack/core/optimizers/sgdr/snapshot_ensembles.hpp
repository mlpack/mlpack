/**
 * @file snapshot_ensembles.hpp
 * @author Marcus Edel
 *
 * Definition of the Snapshot ensembles technique described in:
 * "Snapshot ensembles: Train 1, get m for free" by G. Huang et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_SGDR_SNAPSHOT_ENSEMBLES_HPP
#define MLPACK_CORE_OPTIMIZERS_SGDR_SNAPSHOT_ENSEMBLES_HPP

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
 * @inproceedings{Huang2017,
 *   title     = {Snapshot ensembles: Train 1, get m for free},
 *   author    = {Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu,
 *                John E. Hopcroft, and Kilian Q. Weinberger},
 *   booktitle = {Proceedings of the International Conference on Learning
 *                Representations (ICLR)},
 *   year      = {2017},
 *   url       = {https://arxiv.org/abs/1704.00109}
 * }
 * @endcode
 */
class SnapshotEnsembles
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
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param snapshots Maximum number of snapshots.
   */
  SnapshotEnsembles(const size_t epochRestart,
                    const double multFactor,
                    const double stepSize,
                    const size_t maxIterations,
                    const size_t snapshots) :
    epochRestart(epochRestart),
    multFactor(multFactor),
    constStepSize(stepSize),
    nextRestart(epochRestart),
    batchRestart(0),
    epoch(0)
  {
    snapshotEpochs = 0;
    for (size_t i = 0, er = epochRestart, nr = nextRestart;
        i < maxIterations; ++i)
    {
      if (i > nr)
      {
        er *= multFactor;
        nr += er;
        snapshotEpochs++;
      }
    }

    snapshotEpochs = epochRestart * std::pow(multFactor,
        snapshotEpochs - snapshots + 1);
  }

  /**
   * This function is called in each iteration after the policy update.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
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

      // Create a new snapshot.
      if (epochRestart >= snapshotEpochs)
      {
        snapshots.push_back(iterate);
      }

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

  //! Get the snapshots.
  std::vector<arma::mat> Snapshots() const { return snapshots; }
  //! Modify the snapshots.
  std::vector<arma::mat>& Snapshots() { return snapshots; }

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

  //! Epochs where a new snapshot is created.
  size_t snapshotEpochs;

  //! Locally-stored parameter snapshots.
  std::vector<arma::mat> snapshots;
};

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_SGDR_CYCLICAL_DECAY_HPP
