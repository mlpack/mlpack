/**
 * @file constraints_impl.h
 * @author Manish Kumar
 *
 * Implementation of the Constraints class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP
#define MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP

// In case it hasn't been included already.
#include "constraints.hpp"

#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

namespace mlpack {
namespace lmnn {

Constraints::Constraints(
    const arma::mat& dataset,
    const arma::Row<size_t>& labels,
    size_t k) :
    dataset(math::MakeAlias(const_cast<arma::mat&>(dataset), false)),
    labels(math::MakeAlias(const_cast<arma::Row<size_t>&>(labels), false)),
    k(k)
{
  // Ensure a valid k is passed.
  size_t minCount = arma::min(arma::histc(labels, arma::unique(labels)));

  if (minCount < k)
  {
    Log::Fatal << "Constraints::VerifyK(): One of the class contains only "
        << minCount << " instances, but value of k is " << k << "  "
        << "(k should be < " << minCount << ")!" << std::endl;
  }
}

// Calculates k similar labeled nearest neighbors.
void Constraints::TargetNeighbors(arma::Mat<size_t>& outputMatrix)
{
  size_t N = dataset.n_cols;

  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  outputMatrix = arma::Mat<size_t>(k, N, arma::fill::zeros);

  // KNN instance.
  neighbor::KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec indexSame;

  for ( size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate Target Neighbors.
    indexSame = arma::find(labels == uniqueLabels[i]);

    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame));
    knn.Search(k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexSame.at(neighbors(j));

    // Store target neihbors.
    outputMatrix.cols(indexSame) = neighbors;
  }
}

// Calculates k similar labeled nearest neighbors  on a
// batch of data points.
void Constraints::TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                                  const size_t begin,
                                  const size_t batchSize)
{
  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  // KNN instance.
  neighbor::KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec indexSame;
  arma::uvec subIndexSame;

  for ( size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate Target Neighbors.
    indexSame = arma::find(labels == uniqueLabels[i]);
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexSame.at(neighbors(j));

    // Store target neighbors.
    outputMatrix.cols(begin + subIndexSame) = neighbors;
  }
}

// Calculates k differently labeled nearest neighbors.
void Constraints::Impostors(arma::Mat<size_t>& outputMatrix)
{
  size_t N = dataset.n_cols;

  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  outputMatrix = arma::Mat<size_t>(k, N, arma::fill::zeros);

  // KNN instance.
  neighbor::KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec indexSame;
  arma::uvec indexDiff;

  for ( size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate impostors.
    indexSame = arma::find(labels == uniqueLabels[i]);
    indexDiff = arma::find(labels != uniqueLabels[i]);

    // Perform KNN search with differently labeled points as reference
    // set and  same class points as query set.
    knn.Train(dataset.cols(indexDiff));
    knn.Search(dataset.cols(indexSame), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexDiff.at(neighbors(j));

    // Store impostors.
    outputMatrix.cols(indexSame) =  neighbors;
  }
}

// Calculates k differently labeled nearest neighbors on a
// batch of data points.
void Constraints::Impostors(arma::Mat<size_t>& outputMatrix,
                            const size_t begin,
                            const size_t batchSize)
{
  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  // KNN instance.
  neighbor::KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec indexSame;
  arma::uvec indexDiff;

  for ( size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate impostors.
    indexSame = arma::find(sublabels == uniqueLabels[i]);
    indexDiff = arma::find(labels != uniqueLabels[i]);

    // Perform KNN search with differently labeled points as reference
    // set and same class points as query set.
    knn.Train(dataset.cols(indexDiff));
    knn.Search(subDataset.cols(indexSame), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexDiff.at(neighbors(j));

    // Store impostors.
    outputMatrix.cols(begin + indexSame) =  neighbors;
  }
}

// Generates {data point, target neighbors, impostors} triplets using
// TargetNeighbors() and Impostors().
void Constraints::Triplets(arma::Mat<size_t>& outputMatrix)
{
  size_t N = dataset.n_cols;

  arma::Mat<size_t> impostors;
  Impostors(impostors);

  arma::Mat<size_t> targetNeighbors;
  TargetNeighbors(targetNeighbors);

  outputMatrix = arma::Mat<size_t>(3, k * k * N , arma::fill::zeros);

  for (size_t i = 0, r = 0; i < N; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      for (size_t l = 0; l < k; l++, r++)
      {
        // Generate triplets.
        outputMatrix(0, r) = i;
        outputMatrix(1, r) = targetNeighbors(j, i);
        outputMatrix(2, r) = impostors(l, i);
      }
    }
  }
}

} // namespace lmnn
} // namespace mlpack

#endif
