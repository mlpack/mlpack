/**
 * @file methods/tsne/tsne.hpp
 * @author Ranjodh Singh
 *
 * Definition of the TSNE class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_HPP
#define MLPACK_METHODS_TSNE_TSNE_HPP

#include <mlpack/core.hpp>

#include "tsne_methods.hpp"

namespace mlpack
{

/**
 * An implementation of t-Distributed Stochastic Neighbor Embedding (t-SNE).
 *
 * For more information, see these papers:
 *
 * @code
 * @article{maaten2008visualizing,
 *   title={Visualizing data using t-SNE},
 *   author = {van der Maaten, Laurens and Hinton, Geoffrey},
 *   journal={Journal of machine learning research},
 *   volume={9},
 *   pages={2579--2605},
 *   month = {11},
 *   year={2008}
 * }
 *
 * @code
 * @article{maaten2014accelerating,
 *   title = {Accelerating t-SNE using Tree-Based Algorithms},
 *   author = {van der Maaten, Laurens},
 *   journal={Journal of Machine Learning Research},
 *   volume={15},
 *   pages={3221--3245},
 *   month = {01},
 *   year={2015}
 * }
 * @endcode
 *
 * @code
 * @article{maaten2009learning,
 *   title = {Learning a Parametric Embedding by Preserving Local Structure},
 *   author = {van der Maaten, Laurens},
 *   journal = {Journal of Machine Learning Research - Proceedings Track}
 *   volume = {5},
 *   pages = {384-391},
 *   month = {01},
 *   year = {2009},
 * }
 * @endcode
 *
 * @tparam TSNEStrategy Gradient computation strategy. Options are: "exact",
 *        "dual_tree", "barnes_hut". (Default: "barnes_hut").
 */
template <typename TSNEStrategy = BarnesHutTSNE>
class TSNE
{
 public:
  /**
   * Constructor of the TSNE class.
   *
   * @param outputDim Dimensionality of the embedded space. (Default: 2)
   * @param perplexity Perplexity regulates the balance between local and
   *        global structure preservation, typically set between 5 and 50.
   *        (Default: 30.0)
   * @param exaggeration Amplifies pairwise similarities during the initial
   *        optimization phase. This helps form tighter clusters and clearer
   *        separation between them. A higher value increases spacing between
   *        clusters, but if the cost grows during initial iterations consider
   *        reducing this value or lowering the learning rate. (Default: 12.0)
   * @param learningRate Learning rate (step size) for the optimizer. If the 
   *        specified value is zero, the learning rate is computed
   *        as N / exaggeration everytime Embed is called. (Default: 200.0)
   * @param maxIter Maximum number of iterations. (Default: 1000)
   * @param init Initialization method for the output embedding. Supported
   *        options are: "random", "pca". PCA initialization is recommended
   *        because it often improves both speed and quality. (Default: "pca")
   * @param theta Theta regulates the trade-off between speed and accuracy for
   *        "barnes_hut" and "dual_tree" approximations. The optimal value
   *        differs between approximations. (Default: 0.5)
   */
  TSNE(const size_t outputDim = 2,
       const double perplexity = 30.0,
       const double exaggeration = 12.0,
       const double learningRate = 200.0,
       const size_t maxIter = 1000,
       const std::string& init = "pca",
       const double theta = 0.5);

  /**
   * Embed the given data into a lower-dimensional space.
   *
   * @param X The input data. (input_dimensions X N)
   * @param Y The output embedding. (output_dimensions X N)
   */
  template <typename MatType = arma::mat>
  void Embed(const MatType& X, MatType& Y);

  //! Get the number of output dimensions.
  size_t OutputDimensions() const { return outputDim; }
  //! Modify the number of output dimensions.
  size_t& OutputDimensions() { return outputDim; }

  //! Get the perplexity.
  double Perplexity() const { return perplexity; }
  //! Modify the perplexity.
  double& Perplexity() { return perplexity; }

  //! Get the initial exaggeration factor.
  double Exaggeration() const { return exaggeration; }
  //! Modify the initial exaggeration factor.
  double& Exaggeration() { return exaggeration; }

  //! Get the learning rate (step size) used by the optimizer.
  double LearningRate() const { return learningRate; }
  //! Modify the learning rate.
  double& LearningRate() { return learningRate; }

  //! Get the maximum number of iterations.
  size_t MaximumIterations() const { return maxIter; }
  //! Modify the maximum number of iterations.
  size_t& MaximumIterations() { return maxIter; }

  //! Get the initialization method string (e.g. "pca").
  const std::string& Initialization() const { return init; }
  //! Modify the initialization method string.
  std::string& Initialization() { return init; }

  //! Get the theta parameter for approximation.
  double Theta() const { return theta; }
  //! Modify the theta parameter for approximation.
  double& Theta() { return theta; }

 private:
  //! The number of dimensions to embed into (e.g., 2 or 3).
  size_t outputDim;

  //! The perplexity of the Gaussian distribution.
  double perplexity;

  //! Exaggeration applied during the initial optimization phase.
  double exaggeration;

  //! Learning rate (aka step size) for optimization.
  double learningRate;

  //! The maximum number of iterations.
  size_t maxIter;

  //! Initialization method ("pca", "random", ...).
  std::string init;

  //! The coarseness of the approximation.
  double theta;
};

} // namespace mlpack

// Include implementation.
#include "tsne_impl.hpp"

#endif // MLPACK_METHODS_TSNE_TSNE_HPP
