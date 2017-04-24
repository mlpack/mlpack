/**
 * @file sgdr.hpp
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
#ifndef MLPACK_CORE_OPTIMIZERS_SGDR_SGDR_HPP
#define MLPACK_CORE_OPTIMIZERS_SGDR_SGDR_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/momentum_update.hpp>
#include <mlpack/core/optimizers/sgdr/cyclical_decay.hpp>
#include <mlpack/core/optimizers/sgdr/snapshot_ensembles.hpp>

namespace mlpack {
namespace optimization {

/**
 * This class is based on Mini-batch Stochastic Gradient Descent class and
 * simulates a new warm-started run/restart once a number of epochs are
 * performed this class also implements the Snapshot ensembles technique.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Loshchilov2016,
 *   title   = {{SGDR:} Stochastic Gradient Descent with Restarts},
 *   author  = {Ilya Loshchilov and Frank Hutter},
 *   journal = {CoRR},
 *   year    = {2016}
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
 *   year      = {2017}
 * }
 * @endcode
 *
 * @tparam DecomposableFunctionType Decomposable objective function type to be
 *         minimized.
 * @tparam UpdatePolicyType Update policy used during the iterative update
 *         process. By default the vanilla update policy
 *         (see mlpack::optimization::VanillaUpdate) is used.
 * @tparam DecayPolicyType Decay policy used during the iterative update
 *         process to adjust the step size (CyclicalDecay or SnapshotEnsembles).
 */
template<
    typename DecomposableFunctionType,
    typename UpdatePolicyType = MomentumUpdate,
    typename DecayPolicyType = CyclicalDecay
>
class SGDR
{
 public:
  //! Convenience typedef for the internal optimizer construction.
  using OptimizerType = MiniBatchSGDType<
      DecomposableFunctionType, UpdatePolicyType, DecayPolicyType>;

  /**
   * Construct the SGDR optimizer with snapshot ensembles with the given
   * function and parameters. The defaults here are not necessarily good for
   * the given problem, so it is suggested that the values used be tailored for
   * the task at hand.  The maximum number of iterations refers to the maximum
   * number of mini-batches that are processed.
   *
   * @param epochRestart Initial epoch where decay is applied.
   * @param function Function to be optimized (minimized).
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
 template<typename PolicyType = DecayPolicyType>
 SGDR(DecomposableFunctionType& function,
      const size_t epochRestart = 50,
      const double multFactor = 2.0,
      const size_t batchSize = 1000,
      const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const bool shuffle = true,
      const size_t snapshots = 5,
      const bool accumulate = true,
      const UpdatePolicyType& updatePolicy = UpdatePolicyType(),
      const typename std::enable_if_t<std::is_same<
          PolicyType, SnapshotEnsembles>::value>* junk = 0);

  /**
   * Construct the SGDR optimizer with the given function and
   * parameters.  The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored for the task
   * at hand.  The maximum number of iterations refers to the maximum number of
   * mini-batches that are processed.
   *
   * @param epochRestart Initial epoch where decay is applied.
   * @param function Function to be optimized (minimized).
   * @param batchSize Size of each mini-batch.
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the mini-batch order is shuffled; otherwise, each
   *        mini-batch is visited in linear order.
   * @param updatePolicy Instantiated update policy used to adjust the given
   *        parameters.
   */
 template<typename PolicyType = DecayPolicyType>
 SGDR(DecomposableFunctionType& function,
      const size_t epochRestart = 50,
      const double multFactor = 2.0,
      const size_t batchSize = 1000,
      const double stepSize = 0.01,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const bool shuffle = true,
      const UpdatePolicyType& updatePolicy = UpdatePolicyType(),
      const typename std::enable_if_t<std::is_same<
          PolicyType, CyclicalDecay>::value>* junk = 0);

  /**
   * Optimize the given function using SGDR.  The given starting point
   * will be modified to store the finishing point of the algorithm, and the
   * final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @param accumulate Accumulate the snapshot parameter (default true).
   * @return Objective value of the final point.
   */
  template<typename PolicyType = DecayPolicyType>
  double Optimize(arma::mat& iterate,
                  const typename std::enable_if_t<std::is_same<
                      PolicyType, SnapshotEnsembles>::value>* junk = 0);

  /**
   * Optimize the given function using SGDR.  The given starting point
   * will be modified to store the finishing point of the algorithm, and the
   * final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename PolicyType = DecayPolicyType>
  double Optimize(arma::mat& iterate,
                  const typename std::enable_if_t<std::is_same<
                      PolicyType, CyclicalDecay>::value>* junk = 0);

  //! Get the instantiated function to be optimized.
  const DecomposableFunctionType& Function() const
  {
    return optimizer.Function();
  }

  //! Modify the instantiated function.
  DecomposableFunctionType& Function() { return optimizer.Function(); }

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
  template<typename PolicyType = DecayPolicyType>
  typename std::enable_if<
      std::is_same<PolicyType, SnapshotEnsembles>::value,
      std::vector<arma::mat> >::type
  Snapshots() const { return optimizer.DecayPolicy().Snapshots(); }

  //! Modify the snapshots.
  template<typename PolicyType = DecayPolicyType>
  typename std::enable_if<
      std::is_same<PolicyType, SnapshotEnsembles>::value,
      std::vector<arma::mat>& >::type
  Snapshots() { return optimizer.DecayPolicy().Snapshots(); }

  //! Get the snapshots.
  std::vector<arma::mat> Snapshots() const { return junk; }
  //! Modify the snapshots.
  std::vector<arma::mat>& Snapshots() { return junk; }

 private:
  //! The instantiated function.
  DecomposableFunctionType& function;

  //! The size of each mini-batch.
  size_t batchSize;

  //! Whether or not to accumulate the snapshots.
  bool accumulate;

  //! Locally-stored optimizer instance.
  OptimizerType optimizer;

  //! Locally-stored empty snapshots, necessary to provide an output if another
  //! decay policy than SnapshotEnsembles is used.
  std::vector<arma::mat> junk;
};

// Convenience typedef.

/**
 * Stochastic Gradient Descent with Restarts and snapshot ensembles.
 */
template<typename DecomposableFunctionType>
using SnapshotSGDR = SGDR<
    DecomposableFunctionType, MomentumUpdate,SnapshotEnsembles>;

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "sgdr_impl.hpp"

#endif
