/**
 * @file methods/tsne/tsne_impl.hpp
 * @author Ranjodh Singh
 *
 * Implementation of the TSNE class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_IMPL_HPP
#define MLPACK_METHODS_TSNE_TSNE_IMPL_HPP

#include "tsne.hpp"

namespace mlpack {

template <typename TSNEMethod, typename MatType, typename DistanceType>
TSNE<TSNEMethod, MatType, DistanceType>::TSNE(
    const size_t outputDims,
    const double perplexity,
    const double exaggeration,
    const double stepSize,
    const size_t maxIter,
    const double tolerance,
    const std::string& init,
    const double theta)
    : outputDims(outputDims), perplexity(perplexity),
      exaggeration(exaggeration), stepSize(stepSize), maxIter(maxIter),
      tolerance(tolerance), init(init), theta(theta)
{
  // Nothing To Do Here
}

template <typename TSNEMethod, typename MatType, typename DistanceType>
double TSNE<TSNEMethod, MatType, DistanceType>::Embed(
    const MatType& X, MatType& Y)
{
  // TODO: Make Early Exaggeration Iterations a parameter.
  // TODO: Make Mommentum, Kappa, Phi and MinGain parameters.
  // TODO: Switch PrintLoss() with ProgressBar() when its done
  //       and make it only activate when user asks for it (flag -v).

  assert(maxIter >= 250);
  assert(outputDims > 0);
  assert(exaggeration > 0.0);
  assert(perplexity < X.n_cols);

  InitializeEmbedding(X, Y);

  // Automatically choose a good step size.
  // See "Automated optimized parameters for T-distributed stochastic
  // neighbor embedding improve visualization and analysis of large datasets"
  const bool isStepSizeAuto = (stepSize <= 0.0);
  if (isStepSizeAuto)
    stepSize = std::max(50.0, X.n_cols / exaggeration / 4.0);

  // Calculate degrees of freedom.
  // See "Learning a Parametric Embedding by Preserving Local Structure".
  const size_t dof = std::max<size_t>(1, outputDims - 1);

  double finalObjective = std::numeric_limits<double>::infinity();

  TSNEFunction<TSNEMethod, MatType, DistanceType> function(
      X, perplexity, dof, theta);

  // Optimization is done in two phases.
  // See "Visualizing data using t-SNE".
  const size_t exploratoryIters = 250;
  const size_t convergenceIters = maxIter ? maxIter - exploratoryIters
                                          : std::numeric_limits<size_t>::max();

  // Exploratory Phase
  ens::MomentumDeltaBarDelta exploratoryOptimizer(
      stepSize, exploratoryIters, tolerance, 0.2, 0.8, 0.5, 0.01);

  Log::Info << "Starting Exploratory Phase of t-SNE Optimization."
            << std::endl;

  // Start Exaggerating
  function.InputSimilarities() *= exaggeration;

  finalObjective = exploratoryOptimizer.Optimize(function, Y);

  // Stop Exaggerating
  function.InputSimilarities() /= exaggeration;

  Log::Info << "Completed Exploratory Phase of t-SNE Optimization."
            << std::endl;

  // Convergence Phase
  ens::MomentumDeltaBarDelta convergenceOptimizer(
      stepSize, convergenceIters, tolerance, 0.2, 0.8, 0.8, 0.01);

  Log::Info << "Starting Convergence Phase of t-SNE Optimization."
            << std::endl;

  if (convergenceIters)
    finalObjective = convergenceOptimizer.Optimize(function, Y);

  Log::Info << "Completed Convergence Phase of t-SNE Optimization."
            << std::endl;

  Log::Info << "Final Objective: " << finalObjective << std::endl;

  // If the stepSize was set automatically, reset it to
  // zero so that the next call to `Embed()` can recompute it.
  if (isStepSizeAuto)
    stepSize = 0;

  return finalObjective;
}

template <typename TSNEMethod, typename MatType, typename DistanceType>
void TSNE<TSNEMethod, MatType, DistanceType>::InitializeEmbedding(
    const MatType& X, MatType& Y)
{
  if (init == "pca")
  {
    PCA pca;
    pca.Apply(X, Y, outputDims);
    Y /= stddev(Y.row(0));
  }
  else if (init == "random")
  {
    Y.randn(outputDims, X.n_cols);
  }
  else
  {
    throw std::invalid_argument("invalid initialization type");
  }

  Y *= 1e-4;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_IMPL_HPP
