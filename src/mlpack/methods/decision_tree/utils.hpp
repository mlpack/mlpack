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

namespace mlpack {

/**
 * Calculates the weighted sum and total weight of labels.
 */
template<typename VecType, typename WeightVecType>
inline void WeightedSum(const VecType& values,
                        const WeightVecType& weights,
                        const size_t begin,
                        const size_t end,
                        double& accWeights,
                        double& weightedMean)
{
  using VType = typename VecType::elem_type;
  using WType = typename WeightVecType::elem_type;

  WType totalWeights[4] = { 0.0, 0.0, 0.0, 0.0 };
  VType weightedSum[4] = { 0.0, 0.0, 0.0, 0.0 };

  // SIMD loop: sums four elements simultaneously (if the compiler manages
  // to vectorize the loop).
  for (size_t i = begin + 3; i < end; i += 4)
  {
    const WType weight1 = weights[i - 3];
    const WType weight2 = weights[i - 2];
    const WType weight3 = weights[i - 1];
    const WType weight4 = weights[i];

    weightedSum[0] += weight1 * values[i - 3];
    weightedSum[1] += weight2 * values[i - 2];
    weightedSum[2] += weight3 * values[i - 1];
    weightedSum[3] += weight4 * values[i];

    totalWeights[0] += weight1;
    totalWeights[1] += weight2;
    totalWeights[2] += weight3;
    totalWeights[3] += weight4;
  }

  // Handle leftovers.
  if ((end - begin) % 4 == 1)
  {
    const WType weight1 = weights[end - 1];
    weightedSum[0] += weight1 * values[end - 1];
    totalWeights[0] += weight1;
  }
  else if ((end - begin) % 4 == 2)
  {
    const WType weight1 = weights[end - 2];
    const WType weight2 = weights[end - 1];

    weightedSum[0] += weight1 * values[end - 2];
    weightedSum[1] += weight2 * values[end - 1];

    totalWeights[0] += weight1;
    totalWeights[1] += weight2;
  }
  else if ((end - begin) % 4 == 3)
  {
    const WType weight1 = weights[end - 3];
    const WType weight2 = weights[end - 2];
    const WType weight3 = weights[end - 1];

    weightedSum[0] += weight1 * values[end - 3];
    weightedSum[1] += weight2 * values[end - 2];
    weightedSum[2] += weight1 * values[end - 1];

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
template<typename VecType>
inline void Sum(const VecType& values,
                const size_t begin,
                const size_t end,
                double& mean)
{
  typename VecType::elem_type total[4] = { 0.0, 0.0, 0.0, 0.0 };

  // SIMD loop: add counts for four elements simultaneously (if the compiler
  // manages to vectorize the loop).
  for (size_t i = begin + 3; i < end; i += 4)
  {
    total[0] += values[i - 3];
    total[1] += values[i - 2];
    total[2] += values[i - 1];
    total[3] += values[i];
  }

  // Handle leftovers.
  if ((end - begin) % 4 == 1)
  {
    total[0] += values[end - 1];
  }
  else if ((end - begin) % 4 == 2)
  {
    total[0] += values[end - 2];
    total[1] += values[end - 1];
  }
  else if ((end - begin) % 4 == 3)
  {
    total[0] += values[end - 3];
    total[1] += values[end - 2];
    total[2] += values[end - 1];
  }

  total[0] += total[1] + total[2] + total[3];

  mean = total[0];
}

} // namespace mlpack

#endif
