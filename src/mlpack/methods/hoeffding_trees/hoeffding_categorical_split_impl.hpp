/**
 * @file hoeffding_categorical_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implemental of the HoeffdingCategoricalSplit class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_IMPL_HPP

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
template<typename eT>
void HoeffdingCategoricalSplit<FitnessFunction>::Train(eT value,
                                                       const size_t label)
{
  // Add this to our counts.
  // 'value' should be categorical, so we should be able to cast to size_t...
  sufficientStatistics(label, size_t(value))++;
}

template<typename FitnessFunction>
double HoeffdingCategoricalSplit<FitnessFunction>::EvaluateFitnessFunction()
    const
{
  return FitnessFunction::Evaluate(sufficientStatistics);
}

template<typename FitnessFunction>
template<typename StreamingDecisionTreeType>
void HoeffdingCategoricalSplit<FitnessFunction>::CreateChildren(
    std::vector<StreamingDecisionTreeType>& children,
    data::DatasetInfo& datasetInfo,
    SplitInfo& splitInfo)
{
  // We'll make one child for each category.
  children.push_back(StreamingDecisionTreeType(datasetInfo));
  // Create the according SplitInfo object.
  splitInfo = SplitInfo(sufficientStatistics.n_cols);
}

template<typename FitnessFunction>
size_t HoeffdingCategoricalSplit<FitnessFunction>::MajorityClass() const
{
  // Calculate the class that we have seen the most of.
  arma::Col<size_t> classCounts = sum(sufficientStatistics, 1);

  arma::uword maxIndex;
  classCounts.max(maxIndex);

  return size_t(maxIndex);
}

} // namespace tree
} // namespace mlpack

#endif
