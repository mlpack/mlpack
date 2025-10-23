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

#include <cmath>
#include <limits>
#include <armadillo>
#include <mlpack/core/util/log.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/pca.hpp>

#include "tsne.hpp"
#include "tsne_functions/tsne_function.hpp"

namespace mlpack {

template <typename TSNEMethod>
TSNE<TSNEMethod>::TSNE(const size_t outputDims,
                       const double perplexity,
                       const double exaggeration,
                       const double stepSize,
                       const size_t maxIter,
                       const std::string& init,
                       const double theta)
    : outputDims(outputDims), perplexity(perplexity),
      exaggeration(exaggeration), stepSize(stepSize), maxIter(maxIter),
      init(init), theta(theta)
{
  // Nothing To Do Here
}

template <typename TSNEMethod>
template <typename MatType>
double TSNE<TSNEMethod>::Embed(const MatType& X, MatType& Y)
{
  // To Do: Seperate This Mess Into Functions.
  // To Do: Make PrintLoss() activate only when --verbose
  // To Do: Switch PrintLoss() with ProgressBar() when its done.
  // To Do: Make Early Exaggeration Iterations a parameter
  // To Do: Make Mommentum, Kappa, Phi and MinGain parameters

  // Assertions
  assert(exaggeration != 0);

  // Initialize Embeddings
  InitializeEmbedding(X, Y);

  // Optimization is done in two phases.
  // See "Visualizing data using t-SNE".
  const size_t exploratoryIters = std::min(250UL, maxIter);
  const size_t convergenceIters = std::max(0UL, maxIter - exploratoryIters);

  // Calculate degrees of freedom.
  // See "Learning a Parametric Embedding by Preserving Local Structure".
  const size_t dof = std::max(1UL, outputDims - 1);

  // Automatically choose a good step size.
  // See "Automated optimized parameters for T-distributed stochastic
  // neighbor embedding improve visualization and analysis of large datasets"
  const bool isStepSizeAuto = (bool)(stepSize == 0);
  if (isStepSizeAuto)
    stepSize = std::max(200.0, X.n_cols / exaggeration);

  // Store final kl divergence value after optimization.
  double finalObjective = std::numeric_limits<double>::infinity();

  // Initialize Objective Function.
  TSNEFunction<TSNEMethod> function(X, perplexity, dof, theta);


  // Exploratory Phase Optimizer
  ens::DeltaBarDelta exploratoryOptimizer(
      stepSize,
      exploratoryIters,
      1e-12,
      ens::DeltaBarDeltaUpdate(0.2, 0.8, 0.5, 0.01));

  // Exploratory Phase Optimization
  Log::Info << "Starting Exploratory Phase of t-SNE Optimization."
            << std::endl;
  // Start Exaggerating
  function.InputJointProbabilities() *= exaggeration;
  // Optimize
  if (exploratoryIters)
    finalObjective = exploratoryOptimizer.Optimize(
        function, Y);
  // Stop Exaggerating
  function.InputJointProbabilities() /= exaggeration;
  Log::Info << "Completed Exploratory Phase of t-SNE Optimization."
            << std::endl;


  // Convergence Phase Optimizer
  ens::DeltaBarDelta convergenceOptimizer(
      stepSize,
      convergenceIters,
      1e-12,
      ens::DeltaBarDeltaUpdate(0.2, 0.8, 0.8, 0.01));

  // Convergence Phase Optimization
  Log::Info << "Starting Convergence Phase of t-SNE Optimization."
            << std::endl;
  // Optimize
  if (convergenceIters)
    finalObjective = convergenceOptimizer.Optimize(
        function, Y);
  Log::Info << "Completed Convergence Phase of t-SNE Optimization."
            << std::endl;


  // Report the final objective value.
  Log::Info << "Final Objective: " << finalObjective << std::endl;

  // If the stepSize was set automatically, reset it to
  // zero so that the next call to embed can recompute it.
  if (isStepSizeAuto)
    stepSize = 0;

  return finalObjective;
}

template <typename TSNEMethod>
template <typename MatType>
void TSNE<TSNEMethod>::InitializeEmbedding(const MatType &X, MatType &Y)
{
  // To Do: Handle the case in pca where stddev is zero.
  if (init == "pca")
  {
    PCA pca;
    pca.Apply(X, Y, outputDims);
    Y.each_col() /= arma::stddev(Y, 0, 1);
    Y *= 1e-4;
  }
  else if (init == "random")
  {
    Y.randn(outputDims, X.n_cols);
    Y *= 1e-4;
  }
  else
  {
    throw std::invalid_argument("Invalid init type");
  }
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_IMPL_HPP
