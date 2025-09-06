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
#include "tsne_exaggeration.hpp"
#include "tsne_functions/tsne_function.hpp"

namespace mlpack
{

template <typename TSNEPolicy>
TSNE<TSNEPolicy>::TSNE(const size_t outputDim,
                       const double perplexity,
                       const double earlyExaggeration,
                       const double learningRate,
                       const size_t maxIter,
                       const std::string& init,
                       const double theta)
    : outputDim(outputDim), perplexity(perplexity),
      earlyExaggeration(earlyExaggeration), learningRate(learningRate),
      maxIter(maxIter), init(init), theta(theta)
{
  // Nothing To Do Here
}

template <typename TSNEPolicy>
template <typename MatType>
void TSNE<TSNEPolicy>::Embed(const MatType& X, MatType& Y)
{
  // TODO: VERBOSITY(PROGRESSBAR), THROW ERRORS(init, method)
  // TODO: ADD LEARNING PARAMETER CONTROLS VIA CALLBACKS (ROUND OUTPUT)

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
    // THROW ERROR
  }

  // Initialize The Optimizer
  TSNEOptimizer optimizer(learningRate, X.n_cols, maxIter * X.n_cols);

  // Initialize Objective Function
  typename TSNEFunction<TSNEPolicy>::type function(X, perplexity, earlyExaggeration);

  // Call The Optimizer On The Objective Function
  optimizer.Optimize(function, Y, ens::ProgressBar());
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_IMPL_HPP
