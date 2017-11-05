/**
 * @file snapshots_sgdr.hpp
 * @author Marcus Edel
 *
 * Definition of the Stochastic Gradient Descent with Restarts (SGDR) as
 * described in: "SGDR: Stochastic Gradient Descent with Warm Restarts" by
 * I. Loshchilov et al and the Snapshot ensembles technique described in:
 * "Snapshot ensembles: Train 1, get m for free" by G. Huang et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGDR_SNAPSHOT_SGDR_HPP
#define MLPACK_CORE_OPTIMIZERS_SGDR_SNAPSHOT_SGDR_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/momentum_update.hpp>
#include "snapshot_ensembles.hpp"

namespace mlpack {
namespace optimization {

/**
 * This class is based on Mini-batch Stochastic Gradient Descent class and
 * simulates a new warm-started run/restart once a number of epochs are
 * performed using the Snapshot ensembles technique.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Loshchilov2016,
 *   title   = {{SGDR:} Stochastic Gradient Descent with Restarts},
 *   author  = {Ilya Loshchilov and Frank Hutter},
 *   journal = {CoRR},
 *   year    = {2016},
 *   url     = {https://arxiv.org/abs/1608.03983}
 * }
 * @endcode
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
 *
 * @tparam UpdatePolicyType Update policy used during the iterative update
 *         process. By default the momentum update policy
 *         (see mlpack::optimization::MomentumUpdate) is used.
 */
template<typename UpdatePolicyType = MomentumUpdate>
class SnapshotSGDR
{
 public:
  //! Convenience typedef for the internal optimizer construction.
  using OptimizerType = SGD<UpdatePolicyType, SnapshotEnsembles>;

  /**
   * Construct the SnapshotSGDR optimizer with snapshot ensembles with the given
   * function and parameters. The defaults here are not necessarily good for
   * the given problem, so it is suggested that the values used be tailored for
   * the task at hand.  The maximum number of iterations refers to the maximum
   * number of mini-batches that are processed.
   *
   * @param epochRestart Initial epoch where decay is applied.
   * @param batchSize Size of each mini-batch.
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the mini-batch order is shuffled; otherwise, each
   *        mini-batch is visited in linear order.
   * @param snapshots Maximum number of snapshots.
   * @param accumulate Accumulate the snapshot parameter (default true).
   * @param updatePolicy Instantiated update policy used to adjust the given
   *        parameters.
   */
  SnapshotSGDR(const size_t epochRestart = 50,
               const double multFactor = 2.0,
               const size_t batchSize = 1000,
               const double stepSize = 0.01,
               const size_t maxIterations = 100000,
               const double tolerance = 1e-5,
               const bool shuffle = true,
               const size_t snapshots = 5,
               const bool accumulate = true,
               const UpdatePolicyType& updatePolicy = UpdatePolicyType());

  /**
   * Optimize the given function using SGDR.  The given starting point
   * will be modified to store the finishing point of the algorithm, and the
   * final objective value is returned.
   *
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the step size.
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

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

  //! Get the snapshots.
  std::vector<arma::mat> Snapshots() const
  {
    return optimizer.DecayPolicy().Snapshots();
  }
  //! Modify the snapshots.
  std::vector<arma::mat>& Snapshots()
  {
    return optimizer.DecayPolicy().Snapshots();
  }

  //! Get the update policy.
  const UpdatePolicyType& UpdatePolicy() const
  {
    return optimizer.UpdatePolicy();
  }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy()
  {
    return optimizer.UpdatePolicy();
  }

 private:
  //! The size of each mini-batch.
  size_t batchSize;

  //! Whether or not to accumulate the snapshots.
  bool accumulate;

  //! Locally-stored optimizer instance.
  OptimizerType optimizer;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "snapshot_sgdr_impl.hpp"

#endif
