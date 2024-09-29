/**
 * @file methods/decision_tree/fitness_functions/gini_gain.hpp
 * @author Ryan Curtin
 *
 * The GiniGain class, which is a fitness function (FitnessFunction) for
 * decision trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_GINI_GAIN_HPP
#define MLPACK_METHODS_DECISION_TREE_GINI_GAIN_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * The Gini gain, a measure of set purity usable as a fitness function
 * (FitnessFunction) for decision trees.  This is the exact same thing as the
 * well-known Gini impurity, but negated---since the decision tree will be
 * trying to maximize gain (and the Gini impurity would need to be minimized).
 */
class GiniGain
{
 public:
  /**
   * Evaluate the Gini impurity given a vector of class weight counts.
   */
  template<bool UseWeights, typename CountType>
  static double EvaluatePtr(const CountType* counts,
                            const size_t countLength,
                            const CountType totalCount)
  {
    if (totalCount == 0)
      return 0.0;

    CountType impurity = 0.0;
    for (size_t i = 0; i < countLength; ++i)
      impurity += counts[i] * (totalCount - counts[i]);

    return -((double) impurity / ((double) std::pow(totalCount, 2)));
  }

  /**
   * Evaluate the Gini impurity on the given set of labels.  RowType should be
   * an Armadillo vector that holds size_t objects.
   *
   * Note that it is possible that due to floating-point representation issues,
   * it is possible that the gain returned can be very slightly greater than 0!
   * Thus, if you are checking for a perfect fit, be sure to use 'gain >= 0.0'
   * not 'gain == 0.0'.
   *
   * @param labels Set of labels to evaluate Gini impurity on.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weight of labels.
   */
  template<bool UseWeights, typename RowType, typename WeightVecType>
  static double Evaluate(const RowType& labels,
                         const size_t numClasses,
                         const WeightVecType& weights)
  {
    // Corner case: if there are no elements, the impurity is zero.
    if (labels.n_elem == 0)
      return 0.0;

    // Count the number of elements in each class.  Use four auxiliary vectors
    // to exploit SIMD instructions if possible.
    arma::vec countSpace(4 * numClasses);
    arma::vec counts(countSpace.memptr(), numClasses, false, true);
    arma::vec counts2(countSpace.memptr() + numClasses, numClasses, false,
        true);
    arma::vec counts3(countSpace.memptr() + 2 * numClasses, numClasses, false,
        true);
    arma::vec counts4(countSpace.memptr() + 3 * numClasses, numClasses, false,
        true);

    // Calculate the Gini impurity of the un-split node.
    double impurity = 0.0;

    if (UseWeights)
    {
      // Sum all the weights up.
      double accWeights[4] = { 0.0, 0.0, 0.0, 0.0 };

      // SIMD loop: add counts for four elements simultaneously (if the compiler
      // manages to vectorize the loop).
      for (size_t i = 3; i < labels.n_elem; i += 4)
      {
        const double weight1 = weights[i - 3];
        const double weight2 = weights[i - 2];
        const double weight3 = weights[i - 1];
        const double weight4 = weights[i];

        counts[labels[i - 3]] += weight1;
        counts2[labels[i - 2]] += weight2;
        counts3[labels[i - 1]] += weight3;
        counts4[labels[i]] += weight4;

        accWeights[0] += weight1;
        accWeights[1] += weight2;
        accWeights[2] += weight3;
        accWeights[3] += weight4;
      }

      // Handle leftovers.
      if (labels.n_elem % 4 == 1)
      {
        const double weight1 = weights[labels.n_elem - 1];
        counts[labels[labels.n_elem - 1]] += weight1;
        accWeights[0] += weight1;
      }
      else if (labels.n_elem % 4 == 2)
      {
        const double weight1 = weights[labels.n_elem - 2];
        const double weight2 = weights[labels.n_elem - 1];

        counts[labels[labels.n_elem - 2]] += weight1;
        counts2[labels[labels.n_elem - 1]] += weight2;

        accWeights[0] += weight1;
        accWeights[1] += weight2;
      }
      else if (labels.n_elem % 4 == 3)
      {
        const double weight1 = weights[labels.n_elem - 3];
        const double weight2 = weights[labels.n_elem - 2];
        const double weight3 = weights[labels.n_elem - 1];

        counts[labels[labels.n_elem - 3]] += weight1;
        counts2[labels[labels.n_elem - 2]] += weight2;
        counts3[labels[labels.n_elem - 1]] += weight3;

        accWeights[0] += weight1;
        accWeights[1] += weight2;
        accWeights[2] += weight3;
      }

      accWeights[0] += accWeights[1] + accWeights[2] + accWeights[3];
      counts += counts2 + counts3 + counts4;

      // Catch edge case: if there are no weights, the impurity is zero.
      if (accWeights[0] == 0.0)
        return 0.0;

      for (size_t i = 0; i < numClasses; ++i)
      {
        const double f = ((double) counts[i] / (double) accWeights[0]);
        impurity += f * (1.0 - f);
      }
    }
    else
    {
      // SIMD loop: add counts for four elements simultaneously (if the compiler
      // manages to vectorize the loop).
      for (size_t i = 3; i < labels.n_elem; i += 4)
      {
        counts[labels[i - 3]]++;
        counts2[labels[i - 2]]++;
        counts3[labels[i - 1]]++;
        counts4[labels[i]]++;
      }

      // Handle leftovers.
      if (labels.n_elem % 4 == 1)
      {
        counts[labels[labels.n_elem - 1]]++;
      }
      else if (labels.n_elem % 4 == 2)
      {
        counts[labels[labels.n_elem - 2]]++;
        counts2[labels[labels.n_elem - 1]]++;
      }
      else if (labels.n_elem % 4 == 3)
      {
        counts[labels[labels.n_elem - 3]]++;
        counts2[labels[labels.n_elem - 2]]++;
        counts3[labels[labels.n_elem - 1]]++;
      }

      counts += counts2 + counts3 + counts4;

      for (size_t i = 0; i < numClasses; ++i)
      {
        const double f = ((double) counts[i] / (double) labels.n_elem);
        impurity += f * (1.0 - f);
      }
    }

    return -impurity;
  }

  /**
   * Return the range of the Gini impurity for the given number of classes.
   * (That is, the difference between the maximum possible value and the minimum
   * possible value.)
   *
   * @param numClasses Number of classes in the dataset.
   */
  static double Range(const size_t numClasses)
  {
    // The best possible case is that only one class exists, which gives a Gini
    // impurity of 0.  The worst possible case is that the classes are evenly
    // distributed, which gives n * (1/n * (1 - 1/n)) = 1 - 1/n.
    return 1.0 - (1.0 / double(numClasses));
  }
};

} // namespace mlpack

#endif
