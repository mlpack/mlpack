/**
 * @file methods/decision_tree/fitness_functions/information_gain.hpp
 * @author Ryan Curtin
 *
 * An implementation of information gain, which can be used in place of Gini
 * gain.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_INFORMATION_GAIN_HPP
#define MLPACK_METHODS_DECISION_TREE_INFORMATION_GAIN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The standard information gain criterion, used for calculating gain in
 * decision trees.
 */
class InformationGain
{
 public:
  /**
   * Evaluate the information gain given a vector of class weight counts.
   */
  template<bool UseWeights, typename CountType>
  static double EvaluatePtr(const CountType* counts,
                            const size_t countLength,
                            const CountType totalCount)
  {
    double gain = 0.0;

    for (size_t i = 0; i < countLength; ++i)
    {
      const double f = ((double) counts[i] / (double) totalCount);
      if (f > 0.0)
        gain += f * std::log2(f);
    }

    return gain;
  }

  /**
   * Given a set of labels, calculate the information gain of those labels.
   * Note that it is possible that due to floating-point representation issues,
   * it is possible that the gain returned can be very slightly greater than 0!
   * Thus, if you are checking for a perfect fit, be sure to use 'gain >= 0.0'
   * not 'gain == 0.0'.
   *
   * @param labels Labels of the dataset.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights associated with labels.
   */
  template<bool UseWeights, typename WeightsType>
  static double Evaluate(const arma::Row<size_t>& labels,
                         const size_t numClasses,
                         const WeightsType& weights)
  {
     // Edge case: if there are no elements, the gain is zero.
     if (labels.n_elem == 0)
       return 0.0;

    // Calculate the information gain.
    double gain = 0.0;

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

      // Corner case: return 0 if no weight.
      if (accWeights[0] == 0.0)
        return 0.0;

      for (size_t i = 0; i < numClasses; ++i)
      {
        const double f = ((double) counts[i] / (double) accWeights[0]);
        if (f > 0.0)
          gain += f * std::log2(f);
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
        if (f > 0.0)
          gain += f * std::log2(f);
      }
    }

    return gain;
  }

  /**
   * Return the range of the information gain for the given number of classes.
   * (That is, the difference between the maximum possible value and the minimum
   * possible value.)
   *
   * @param numClasses Number of classes in the dataset.
   */
  static double Range(const size_t numClasses)
  {
    // The best possible case gives an information gain of 0.  The worst
    // possible case is even distribution, which gives n * (1/n * log2(1/n)) =
    // log2(1/n) = -log2(n).  So, the range is log2(n).
    return std::log2(numClasses);
  }
};

} // namespace mlpack

#endif
