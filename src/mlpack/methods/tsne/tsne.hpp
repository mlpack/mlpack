/**
 * @file methods/tsne/tsne.hpp
 * @author Kiner Shah
 *
 * Defines the TSNE class to perform t-Distributed Stochastic Neighbor
 * Embedding on the given dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_HPP
#define MLPACK_METHODS_TSNE_TSNE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The TSNE class implements t-Distributed Stochastic Neighbor Embedding,
 * a dimensionality reduction technique that is particularly well suited
 * for the visualization of high-dimensional datasets.
 *
 * t-SNE is a variation of Stochastic Neighbor Embedding that is much easier
 * to optimize, and produces significantly better visualizations by reducing
 * the tendency to crowd points together in the center of the map.
 *
 * For more details, see:
 * - Van der Maaten, L. and Hinton, G. "Visualizing Data using t-SNE",
 *   Journal of Machine Learning Research 9(Nov):2579-2605, 2008.
 *
 * Example usage:
 *
 * @code
 * arma::mat data; // Load data
 * TSNE tsne(30.0, 200.0, 1000); // perplexity=30, learningRate=200, maxIter=1000
 * arma::mat output;
 * tsne.Apply(data, output, 2); // Reduce to 2 dimensions
 * @endcode
 */
template<typename MatType = arma::mat>
class TSNE
{
 public:
  /**
   * Create the TSNE object with the given parameters.
   *
   * @param perplexity The perplexity value for the Gaussian kernel in the
   *     high-dimensional space. A larger value gives more focus on global
   *     structure. Typical values are between 5 and 50.
   * @param learningRate The learning rate for gradient descent.
   * @param maxIterations Maximum number of iterations for optimization.
   * @param earlyExaggeration Coefficient for early exaggeration. During the
   *     first iterations, the P values are multiplied by this value to allow
   *     clusters to move around more freely.
   */
  TSNE(const double perplexity = 30.0,
       const double learningRate = 200.0,
       const size_t maxIterations = 1000,
       const double earlyExaggeration = 12.0);

  /**
   * Apply t-SNE to the provided data set, reducing it to the specified
   * number of dimensions.
   *
   * @param data Input dataset (each column is a point).
   * @param output Output low-dimensional embedding (each column is a point).
   * @param newDimension Desired dimensionality of output (default: 2).
   */
  template<typename OutMatType = MatType>
  void Apply(const MatType& data,
             OutMatType& output,
             const size_t newDimension = 2);

  //! Get the perplexity.
  double Perplexity() const { return perplexity; }
  //! Modify the perplexity.
  double& Perplexity() { return perplexity; }

  //! Get the learning rate.
  double LearningRate() const { return learningRate; }
  //! Modify the learning rate.
  double& LearningRate() { return learningRate; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the early exaggeration coefficient.
  double EarlyExaggeration() const { return earlyExaggeration; }
  //! Modify the early exaggeration coefficient.
  double& EarlyExaggeration() { return earlyExaggeration; }

 private:
  //! Perplexity for Gaussian kernel.
  double perplexity;
  //! Learning rate for gradient descent.
  double learningRate;
  //! Maximum number of iterations.
  size_t maxIterations;
  //! Early exaggeration coefficient.
  double earlyExaggeration;

  /**
   * Compute pairwise affinities in the high-dimensional space using
   * a Gaussian kernel with perplexity-based precision.
   *
   * @param data Input data matrix.
   * @param affinities Output affinity matrix P.
   */
  void ComputeAffinities(const MatType& data, MatType& affinities);

  /**
   * Perform binary search to find the precision (beta = 1 / (2 * sigma^2))
   * that results in the desired perplexity for a given point.
   *
   * @param distances Distances from one point to all others.
   * @param perplexity Desired perplexity.
   * @param probabilities Output conditional probabilities.
   */
  void SearchPrecision(const typename MatType::col_type& distances,
                       const double perplexity,
                       typename MatType::col_type& probabilities);

  /**
   * Compute the gradient of the t-SNE cost function.
   *
   * @param P High-dimensional affinities.
   * @param Y Current low-dimensional embedding.
   * @param gradient Output gradient matrix.
   */
  void ComputeGradient(const MatType& P,
                       const MatType& Y,
                       MatType& gradient);
};

} // namespace mlpack

// Include implementation.
#include "tsne_impl.hpp"

#endif
