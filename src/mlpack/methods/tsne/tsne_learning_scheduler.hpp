/**
 * @file methods/tsne/tsne_learning_scheduler.hpp
 * @author Ranjodh Singh
 *
 * A callback used by the default TSNE optimizer for Switching B/W
 * Exploratory and Convergence Phases during TSNE optimization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_LEARNING_SHEDULER_HPP
#define MLPACK_METHODS_TSNE_TSNE_LEARNING_SHEDULER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack
{

class TSNELearningScheduler
{
 public:
  TSNELearningScheduler(const size_t earlyIters,
                       const double exaggeration,
                       const double initMomentum,
                       const double finalMomentum)
      : earlyIters(earlyIters), exaggeration(exaggeration),
        initMomentum(initMomentum), finalMomentum(finalMomentum)
  {
    /* Nothing To Do Here */
  }

  template <typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType& optimizer,
                         FunctionType& function,
                         MatType& /* coordinates */)
  {
    function.InputJointProbabilities() *= exaggeration;
    optimizer.UpdatePolicy().Momentum() = initMomentum;
  }

  template <typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& optimizer,
                FunctionType& function,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double /* objective */)
  {
    if (epoch == earlyIters)
    {
      function.InputJointProbabilities() /= exaggeration;
      optimizer.UpdatePolicy().Momentum() = finalMomentum;
    }
    return false;
  }

 private:
  const size_t earlyIters;
  const double exaggeration;
  const double initMomentum;
  const double finalMomentum;
};

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_LEARNING_SHEDULER_HPP
