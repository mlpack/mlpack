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

#include <armadillo>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/pca.hpp>

#include "tsne.hpp"
#include "tsne_functions/tsne_function.hpp"

namespace mlpack {

template <typename TSNEMethod>
TSNE<TSNEMethod>::TSNE(const size_t outputDim,
                       const double perplexity,
                       const double exaggeration,
                       const double learningRate,
                       const size_t maxIter,
                       const std::string& init,
                       const double theta)
    : outputDim(outputDim), perplexity(perplexity), exaggeration(exaggeration),
      learningRate(learningRate), maxIter(maxIter), init(init), theta(theta)
{
  // Nothing To Do Here
}

template <typename TSNEMethod>
template <typename MatType>
void TSNE<TSNEMethod>::Embed(const MatType& X, MatType& Y)
{
  // Can lead to division by zero
  assert(exaggeration != 0);

  // To Do: Seperate Functions
  // initialize embeddigs and initialize objective function.

  // Initialize Embeddings
  // Its better if the initialization has stddev of 0.0001.
  // See "The art of using t-SNE for single-cell transcriptomics".
  if (init == "pca")
  {
    PCA pca;
    pca.Apply(X, Y, outputDim);

    // To Do: Handle the case where stddev is zero.
    Y.each_col() /= arma::stddev(Y, 0, 1);
    Y *= 1e-4;
  }
  else if (init == "random")
  {
    Y.randn(outputDim, X.n_cols);
    Y *= 1e-4;
  }
  else
  {
    throw std::invalid_argument("invalid init type");
  }

  // Calculate degrees of freedom
  // See "Learning a Parametric Embedding by Preserving Local Structure"
  const size_t dof = std::max<size_t>(Y.n_rows - 1, 1);
  // Initialize Objective Function
  TSNEFunction<TSNEMethod> function(X, perplexity, dof, theta);

  // Automatically choose a good learning rate.
  // See "Automated optimized parameters for T-distributed stochastic
  // neighbor embedding improve visualization and analysis of large datasets"
  const bool isLearningRateAuto = (bool)(learningRate == 0);
  if (isLearningRateAuto)
  {
    learningRate = std::max(200.0, X.n_cols / exaggeration);
  }

  // Call The Optimizer On The Objective Function
  // To Do: Make Early Exaggeration Iterations a parameter
  const size_t exploratoryIters = std::min<size_t>(250, maxIter);
  const size_t convergenceIters = std::max<size_t>(0,
      maxIter - exploratoryIters);

  // Exploratory Phase
  // To Do: Make Mommentum, Kappa, Phi and MinGain parameters
  ens::DeltaBarDelta exploratoryOptimizer(
      learningRate,
      exploratoryIters,
      1e-12,
      ens::DeltaBarDeltaUpdate(0.2, 0.8, 0.5, 0.01));

  Log::Info << "Starting Exploratory Phase of t-SNE Optimization."
            << std::endl;
  // Start Exaggerating
  function.InputJointProbabilities() *= exaggeration;
  // Optimize
  exploratoryOptimizer.Optimize(function, Y, ens::PrintLoss());
  // Stop Exaggerating
  function.InputJointProbabilities() /= exaggeration;
  Log::Info << "Completed Exploratory Phase of t-SNE Optimization."
            << std::endl;

  // Convergence Phase
  ens::DeltaBarDelta convergenceOptimizer(
      learningRate,
      convergenceIters,
      1e-12,
      ens::DeltaBarDeltaUpdate(0.2, 0.8, 0.8, 0.01));

  Log::Info << "Starting Convergence Phase of t-SNE Optimization."
            << std::endl;
  // Optimize
  convergenceOptimizer.Optimize(function, Y, ens::PrintLoss());
  Log::Info << "Completed Convergence Phase of t-SNE Optimization."
            << std::endl;

  // If the learningRate was set automatically, reset it to
  // zero so that the next call to embed can recompute it.
  if (isLearningRateAuto)
  {
    learningRate = 0;
  }
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_IMPL_HPP
