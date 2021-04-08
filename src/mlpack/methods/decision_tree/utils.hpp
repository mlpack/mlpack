/**
 * @file methods/decision_tree/utils.hpp
 * @author Rishabh Garg
 *
 * Various utility functions used in decision tree implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_UTILS_HPP
#define MLPACK_METHODS_DECISION_TREE_UTILS_HPP

/**
 * Calculates the weighted sum and total weight of labels.
 */
void WeightedSum(const arma::rowvec& labels,
                 const arma::rowvec& weights,
                 const size_t begin,
                 const size_t end,
                 double& accWeights,
                 double& weightedMean)
{
  double totalWeights[4] = { 0.0, 0.0, 0.0, 0.0 };
  double weightedSum[4] = { 0.0, 0.0, 0.0, 0.0 };

  // SIMD loop: sums four elements simultaneously (if the compiler manages
  // to vectorize the loop).
  for (size_t i = begin + 3; i < end; i += 4)
  {
    const double weight1 = weights[i - 3];
    const double weight2 = weights[i - 2];
    const double weight3 = weights[i - 1];
    const double weight4 = weights[i];

    weightedSum[0] += weight1 * labels[i - 3];
    weightedSum[1] += weight2 * labels[i - 2];
    weightedSum[2] += weight3 * labels[i - 1];
    weightedSum[3] += weight4 * labels[i];

    totalWeights[0] += weight1;
    totalWeights[1] += weight2;
    totalWeights[2] += weight3;
    totalWeights[3] += weight4;
  }

  // Handle leftovers.
  if ((end - begin) % 4 == 1)
  {
    const double weight1 = weights[end - 1];
    weightedSum[0] += weight1 * labels[end - 1];
    totalWeights[0] += weight1;
  }
  else if ((end - begin) % 4 == 2)
  {
    const double weight1 = weights[end - 2];
    const double weight2 = weights[end - 1];

    weightedSum[0] += weight1 * labels[end - 2];
    weightedSum[1] += weight2 * labels[end - 1];

    totalWeights[0] += weight1;
    totalWeights[1] += weight2;
  }
  else if ((end - begin) % 4 == 3)
  {
    const double weight1 = weights[end - 3];
    const double weight2 = weights[end - 2];
    const double weight3 = weights[end - 1];

    weightedSum[0] += weight1 * labels[end - 3];
    weightedSum[1] += weight2 * labels[end - 2];
    weightedSum[2] += weight1 * labels[end - 1];

    totalWeights[0] += weight1;
    totalWeights[1] += weight2;
    totalWeights[2] += weight3;
  }

  totalWeights[0] += totalWeights[1] + totalWeights[2] + totalWeights[3];
  weightedSum[0] += weightedSum[1] + weightedSum[2] + weightedSum[3];

  accWeights = totalWeights[0];
  weightedMean = weightedSum[0];
}

/**
 * Sums up the labels vector.
 */
void Sum(const arma::rowvec& labels,
        const size_t begin,
        const size_t end,
        double& mean)
{
  double total[4] = { 0.0, 0.0, 0.0, 0.0 };

  // SIMD loop: add counts for four elements simultaneously (if the compiler
  // manages to vectorize the loop).
  for (size_t i = begin + 3; i < end; i += 4)
  {
    total[0] += labels[i - 3];
    total[1] += labels[i - 2];
    total[2] += labels[i - 1];
    total[3] += labels[i];
  }

  // Handle leftovers.
  if (labels.n_elem % 4 == 1)
  {
    total[0] += labels[end - 1];
  }
  else if (labels.n_elem % 4 == 2)
  {
    total[0] += labels[end - 2];
    total[1] += labels[end - 1];
  }
  else if (labels.n_elem % 4 == 3)
  {
    total[0] += labels[end - 3];
    total[1] += labels[end - 2];
    total[2] += labels[end - 1];
  }

  total[0] += total[1] + total[2] + total[3];

  mean = total[0];
}

#endif
