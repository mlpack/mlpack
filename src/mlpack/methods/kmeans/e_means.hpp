/**
 * @file methods/kmeans/e_means.hpp
 *
 * An implementation of e-means, mini-batch k-means with exponential moving
 * averages of cluster counts and sums.  This is a port of the reference
 * implementation in Faiss (faiss/contrib/e_means, MIT license, Copyright (c)
 * Meta Platforms, Inc. and affiliates).
 *
 * @code
 * @misc{bessa2026emeans,
 *   title  = {{e-means}: outperforming {k-means} (in one-tenth the epochs)},
 *   author = {Bessa, Swann and Mazar{\'e}, Pierre-Emmanuel and Szilvasy,
 *             Gergely and Touvron, Hugo and Picard, David and Douze, Matthijs
 *             and J{\'e}gou, Herv{\'e}},
 *   year   = {2026},
 *   note   = {arXiv:2611.09501}
 * }
 * @endcode
 *
 * Parts of this file are ported from Faiss (faiss/contrib/e_means), which is
 * distributed under the following license:
 *
 *   MIT License
 *
 *   Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a
 *   copy of this software and associated documentation files (the "Software"),
 *   to deal in the Software without restriction, including without limitation
 *   the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *   and/or sell copies of the Software, and to permit persons to whom the
 *   Software is furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS IN THE SOFTWARE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_E_MEANS_HPP
#define MLPACK_METHODS_KMEANS_E_MEANS_HPP

#include <mlpack/core.hpp>

#include "sample_initialization.hpp"

namespace mlpack {

/**
 * e-means clustering.  e-means minimizes the k-means objective (the sum of
 * squared Euclidean distances from points to their closest centroid) with
 * mini-batch updates.  Each cluster j keeps an exponential moving average of
 * the number N_j and the sum S_j of the points assigned to it, and its
 * centroid is S_j / N_j.  For a batch in which n_j points with sum s_j are
 * assigned to cluster j,
 *
 *   N_j <- (1 - alpha) N_j + alpha n_j,   S_j <- (1 - alpha) S_j + alpha s_j.
 *
 * The step alpha follows a cosine schedule that ends at one tenth of its
 * initial value alpha0.  The batch size and alpha0 are set by scaling laws
 * fitted by the authors, so the defaults need no tuning.  Every 5 batches,
 * clusters whose mass is below a fraction of the mean mass are revived by
 * splitting a cluster sampled in proportion to its mass.  The returned
 * centroids are the means of the points assigned to each cluster during the
 * last epoch.
 *
 * On high-dimensional data with many clusters, e-means typically reaches the
 * error of 25 Lloyd iterations in 2 to 3 epochs, where each epoch costs about
 * as much as one Lloyd iteration.
 *
 * @tparam InitialPartitionPolicy Initial partitioning policy; it must
 *     implement Cluster(data, clusters, centroids).
 */
template<typename InitialPartitionPolicy = SampleInitialization>
class EMeans
{
 public:
  //! Scaling laws that can set the batch size.
  enum BatchLaw
  {
    //! QUALITY if the dimensionality is at most 16, THROUGHPUT otherwise.
    AUTO,
    //! Larger batches: fewer, cheaper updates.
    THROUGHPUT,
    //! Batches about 8 times smaller than THROUGHPUT.
    QUALITY
  };

  /**
   * Create the EMeans object.
   *
   * @param epochs Number of passes over the data.  A fractional number of
   *     epochs runs a partial pass before the last one, which is always full.
   * @param batchSize Mini-batch size; 0 sets it with the batch law.
   * @param alpha0 Initial step in (0, 1]; 0 sets it with the step law.
   * @param batchLaw Scaling law used to set the batch size if it is 0.
   * @param unusedThreshold Clusters with mass below this fraction of the mean
   *     mass are revived; 0 disables revival.  A negative value selects 0.1,
   *     or 0 if the dimensionality is at most 2.
   * @param partitioner Instantiated initial partitioning policy.
   */
  EMeans(const double epochs = 5.0,
         const size_t batchSize = 0,
         const double alpha0 = 0.0,
         const BatchLaw batchLaw = AUTO,
         const double unusedThreshold = -1.0,
         const InitialPartitionPolicy& partitioner = InitialPartitionPolicy());

  /**
   * Cluster the data, and return the centroids.
   *
   * @param data Dataset to cluster, one point per column.
   * @param clusters Number of clusters.
   * @param centroids Matrix to store the centroids in; if initialGuess is
   *     true, it must hold the initial centroids.
   * @param initialGuess Whether centroids holds the initial centroids.
   */
  template<typename MatType>
  void Cluster(const MatType& data,
               const size_t clusters,
               arma::Mat<typename MatType::elem_type>& centroids,
               const bool initialGuess = false);

  /**
   * Cluster the data, and return the centroids and the assignment of every
   * point to its closest final centroid (this costs one extra pass).
   *
   * @param data Dataset to cluster, one point per column.
   * @param clusters Number of clusters.
   * @param assignments Vector to store the cluster of each point in.
   * @param centroids Matrix to store the centroids in; if initialGuess is
   *     true, it must hold the initial centroids.
   * @param initialGuess Whether centroids holds the initial centroids.
   */
  template<typename MatType>
  void Cluster(const MatType& data,
               const size_t clusters,
               arma::Row<size_t>& assignments,
               arma::Mat<typename MatType::elem_type>& centroids,
               const bool initialGuess = false);

  /**
   * Assign every point to its closest centroid.
   *
   * @param data Points, one per column.
   * @param centroids Centroids, one per column.
   * @param assignments Vector to store the closest centroid of each point in.
   * @param distances If not null, vector to store the squared distance of
   *     each point to its closest centroid in.
   */
  template<typename MatType>
  static void Assign(const MatType& data,
                     const arma::Mat<typename MatType::elem_type>& centroids,
                     arma::Row<size_t>& assignments,
                     arma::Col<typename MatType::elem_type>* distances =
                         nullptr);

  /**
   * Lower-level interface: run e-means from the given centroids with a given
   * batch size and initial step, drawing the order of the points in each
   * epoch and the uniform variates used for revival from the given
   * functions.  Most users want Cluster() instead.
   *
   * @param data Dataset to cluster, one point per column.
   * @param centroids Initial centroids on input, final centroids on output.
   * @param batchSize Mini-batch size.
   * @param alpha0 Initial step.
   * @param epochs Number of epochs.
   * @param unusedThreshold Revival threshold (0 disables revival).
   * @param epochOrder Function that takes an epoch index and returns an
   *     arma::uvec permutation of the point indices.
   * @param uniform Function that takes a size k and returns an arma::vec of k
   *     uniform variates in [0, 1).
   * @param steps Set to the number of batch updates.
   * @param splits Set to the number of revived clusters.
   */
  template<typename MatType, typename OrderFunction, typename UniformFunction>
  static void Train(const MatType& data,
                    arma::Mat<typename MatType::elem_type>& centroids,
                    const size_t batchSize,
                    const double alpha0,
                    const double epochs,
                    const double unusedThreshold,
                    OrderFunction&& epochOrder,
                    UniformFunction&& uniform,
                    size_t& steps,
                    size_t& splits);

  /**
   * Batch size given by a batch law, for n points, k clusters, the given
   * number of epochs and dimensionality d.
   */
  static size_t BatchSizeLaw(const size_t n,
                             const size_t k,
                             const double epochs,
                             const size_t d,
                             const BatchLaw law = AUTO);

  //! Initial step given by the step law for the given batch size.
  static double StepLaw(const size_t n,
                        const size_t k,
                        const double epochs,
                        const size_t batchSize);

  //! Step used for batch `step` out of `totalSteps` (cosine schedule).
  static double ScheduledAlpha(const double alpha0,
                               const size_t step,
                               const size_t totalSteps);

  //! Number of points visited in each epoch.
  static std::vector<size_t> EpochRows(const size_t n, const double epochs);

  //! Get the number of epochs.
  double Epochs() const { return epochs; }
  //! Modify the number of epochs.
  double& Epochs() { return epochs; }

  //! Get the batch size (0 means it is set by the batch law).
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size (0 means it is set by the batch law).
  size_t& BatchSize() { return batchSize; }

  //! Get the initial step (0 means it is set by the step law).
  double Alpha0() const { return alpha0; }
  //! Modify the initial step (0 means it is set by the step law).
  double& Alpha0() { return alpha0; }

  //! Get the batch law.
  BatchLaw Law() const { return batchLaw; }
  //! Modify the batch law.
  BatchLaw& Law() { return batchLaw; }

  //! Get the revival threshold (negative means it depends on dimensionality).
  double UnusedThreshold() const { return unusedThreshold; }
  //! Modify the revival threshold (negative means it depends on
  //! dimensionality).
  double& UnusedThreshold() { return unusedThreshold; }

  //! Get the initial partitioning policy.
  const InitialPartitionPolicy& Partitioner() const { return partitioner; }
  //! Modify the initial partitioning policy.
  InitialPartitionPolicy& Partitioner() { return partitioner; }

  //! Batch size used during the last call to Cluster().
  size_t UsedBatchSize() const { return usedBatchSize; }
  //! Initial step used during the last call to Cluster().
  double UsedAlpha0() const { return usedAlpha0; }
  //! Number of batch updates during the last call to Cluster().
  size_t Steps() const { return steps; }
  //! Number of clusters revived during the last call to Cluster().
  size_t Splits() const { return splits; }

  //! Serialize the object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Number of epochs.
  double epochs;
  //! Batch size, or 0 for the batch law.
  size_t batchSize;
  //! Initial step, or 0 for the step law.
  double alpha0;
  //! Batch law.
  BatchLaw batchLaw;
  //! Revival threshold, or negative for the default.
  double unusedThreshold;
  //! Instantiated initial partitioning policy.
  InitialPartitionPolicy partitioner;

  //! Batch size used during the last call to Cluster().
  size_t usedBatchSize;
  //! Initial step used during the last call to Cluster().
  double usedAlpha0;
  //! Number of batch updates during the last call to Cluster().
  size_t steps;
  //! Number of revived clusters during the last call to Cluster().
  size_t splits;
};

} // namespace mlpack

// Include implementation.
#include "e_means_impl.hpp"

#endif
