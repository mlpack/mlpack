/**
 * @file hoeffding_categorical_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implemental of the HoeffdingCategoricalSplit class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_IMPL_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_IMPL_HPP

// In case it hasn't been included yet.
#include "hoeffding_categorical_split.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
HoeffdingCategoricalSplit<FitnessFunction>::HoeffdingCategoricalSplit(
    const size_t numCategories,
    const size_t numClasses) :
    sufficientStatistics(numClasses, numCategories)
{
  sufficientStatistics.zeros();
}

template<typename FitnessFunction>
HoeffdingCategoricalSplit<FitnessFunction>::HoeffdingCategoricalSplit(
    const size_t numCategories,
    const size_t numClasses,
    const HoeffdingCategoricalSplit& /* other */) :
    sufficientStatistics(numClasses, numCategories)
{
  sufficientStatistics.zeros();
}

template<typename FitnessFunction>
template<typename eT>
void HoeffdingCategoricalSplit<FitnessFunction>::Train(eT value,
                                                       const size_t label)
{
  // Add this to our counts.
  // 'value' should be categorical, so we should be able to cast to size_t...
  sufficientStatistics(label, size_t(value))++;
}

template<typename FitnessFunction>
void HoeffdingCategoricalSplit<FitnessFunction>::EvaluateFitnessFunction(
    double& bestFitness,
    double& secondBestFitness) const
{
  bestFitness = FitnessFunction::Evaluate(sufficientStatistics);
  secondBestFitness = 0.0; // We only split one possible way.
}

template<typename FitnessFunction>
void HoeffdingCategoricalSplit<FitnessFunction>::Split(
    arma::Col<size_t>& childMajorities,
    SplitInfo& splitInfo)
{
  // We'll make one child for each category.
  childMajorities.set_size(sufficientStatistics.n_cols);
  for (size_t i = 0; i < sufficientStatistics.n_cols; ++i)
  {
    arma::uword maxIndex = 0;
    sufficientStatistics.unsafe_col(i).max(maxIndex);
    childMajorities[i] = size_t(maxIndex);
  }

  // Create the according SplitInfo object.
  splitInfo = SplitInfo(sufficientStatistics.n_cols);
}

template<typename FitnessFunction>
size_t HoeffdingCategoricalSplit<FitnessFunction>::MajorityClass() const
{
  // Calculate the class that we have seen the most of.
  arma::Col<size_t> classCounts = arma::sum(sufficientStatistics, 1);

  arma::uword maxIndex = 0;
  classCounts.max(maxIndex);

  return size_t(maxIndex);
}

template<typename FitnessFunction>
double HoeffdingCategoricalSplit<FitnessFunction>::MajorityProbability() const
{
  arma::Col<size_t> classCounts = arma::sum(sufficientStatistics, 1);

  return double(classCounts.max()) / double(arma::accu(classCounts));
}

} // namespace tree
} // namespace mlpack

#endif
