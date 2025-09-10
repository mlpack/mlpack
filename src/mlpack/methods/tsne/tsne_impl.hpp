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

#include <ensmallen.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/pca.hpp>

#include "tsne.hpp"
#include "tsne_optimizer.hpp"
#include "tsne_functions/tsne_function.hpp"

namespace mlpack
{

template <typename TSNEStrategy>
TSNE<TSNEStrategy>::TSNE(const size_t outputDim,
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

template <typename TSNEStrategy>
template <typename MatType>
void TSNE<TSNEStrategy>::Embed(const MatType& X, MatType& Y)
{
  // To Do: VERBOSITY(PROGRESSBAR)

  // Initialize Embedding
  if (init == "pca")
  {
    PCA pca;
    pca.Apply(X, Y, outputDim);
  }
  else if (init == "random")
  {
    Y.randn(outputDim, X.n_cols);
  }
  else
  {
    /* To Do: Throw Error */
  }

  // Initialize Objective Function
  TSNEFunction<TSNEStrategy> function(X, perplexity);

  // Call The Optimizer On The Objective Function
  const size_t exploratoryIter = std::min<size_t>(250, maxIter);
  const size_t convergenceIter = std::max<size_t>(0,
                                                  maxIter - exploratoryIter);

  TSNEOptimizer exploratoryOptimizer(learningRate,
                                     X.n_cols,
                                     exploratoryIter * X.n_cols);
  TSNEOptimizer convergenceOptimizer(learningRate,
                                     X.n_cols,
                                     convergenceIter * X.n_cols);

  Log::Info << "Starting Exploratory Phase of t-SNE Optimization."
            << std::endl;
  exploratoryOptimizer.Optimize(function, Y, ens::ProgressBar());
  Log::Info << "Completed Exploratory Phase of t-SNE Optimization."
            << std::endl;

  Log::Info << "Starting Convergence Phase of t-SNE Optimization."
            << std::endl;
  convergenceOptimizer.Optimize(function, Y, ens::ProgressBar());
  Log::Info << "Completed Convergence Phase of t-SNE Optimization."
            << std::endl;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_IMPL_HPP
