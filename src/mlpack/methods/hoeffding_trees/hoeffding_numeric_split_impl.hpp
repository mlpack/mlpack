/**
 * @file hoeffding_numeric_split_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of the simple HoeffdingNumericSplit class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_IMPL_HPP

#include "hoeffding_numeric_split.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction, typename ObservationType>
HoeffdingNumericSplit<FitnessFunction, ObservationType>::HoeffdingNumericSplit(
    const size_t numClasses,
    const size_t bins,
    const size_t observationsBeforeBinning) :
    observations(observationsBeforeBinning - 1),
    labels(observationsBeforeBinning - 1),
    bins(bins),
    observationsBeforeBinning(observationsBeforeBinning),
    samplesSeen(0),
    sufficientStatistics(arma::zeros<arma::Mat<size_t>>(numClasses, bins))
{
  // Nothing to do.
}

template<typename FitnessFunction, typename ObservationType>
void HoeffdingNumericSplit<FitnessFunction, ObservationType>::Train(
    ObservationType value,
    const size_t label)
{
  if (samplesSeen < observationsBeforeBinning - 1)
  {
    // Add this to the samples we have seen.
    observations[samplesSeen] = value;
    labels[samplesSeen] = label;
    ++samplesSeen;
    return;
  }
  else if (samplesSeen == observationsBeforeBinning - 1)
  {
    // Now we need to make the bins.
    ObservationType min = value;
    ObservationType max = value;
    for (size_t i = 0; i < observationsBeforeBinning - 1; ++i)
    {
      if (observations[i] < min)
        min = observations[i];
      else if (observations[i] > max)
        max = observations[i];
    }

    // Now split these.  We can't use linspace, because we don't want to include
    // the endpoints.
    splitPoints.resize(bins - 1);
    const ObservationType binWidth = (max - min) / bins;
    for (size_t i = 0; i < bins - 1; ++i)
      splitPoints[i] = min + (i + 1) * binWidth;
    ++samplesSeen;

    // Now, add all of the points we've seen to the sufficient statistics.
    for (size_t i = 0; i < observationsBeforeBinning - 1; ++i)
    {
      // What bin does the point fall into?
      size_t bin = 0;
      while (observations[i] > splitPoints[bin] && bin < bins - 1)
        ++bin;

      sufficientStatistics(labels[i], bin)++;
    }
  }

  // If we've gotten to here, then we need to add the point to the sufficient
  // statistics.  What bin does the point fall into?
  size_t bin = 0;
  while (value > splitPoints[bin] && bin < bins - 1)
    ++bin;

  sufficientStatistics(label, bin)++;
}

template<typename FitnessFunction, typename ObservationType>
double HoeffdingNumericSplit<FitnessFunction, ObservationType>::
    EvaluateFitnessFunction() const
{
  Log::Debug << sufficientStatistics.t();
  if (samplesSeen < observationsBeforeBinning)
    return 0.0;
  else
    return FitnessFunction::Evaluate(sufficientStatistics);
}

template<typename FitnessFunction, typename ObservationType>
template<typename StreamingDecisionTreeType>
void HoeffdingNumericSplit<FitnessFunction, ObservationType>::CreateChildren(
    std::vector<StreamingDecisionTreeType>& children,
    const data::DatasetInfo& datasetInfo,
    const size_t dimensionality,
    SplitInfo& splitInfo)
{
  // We'll make one child for each bin.
  for (size_t i = 0; i < sufficientStatistics.n_cols; ++i)
    children.push_back(StreamingDecisionTreeType(datasetInfo, dimensionality,
        sufficientStatistics.n_rows));

  // Create the SplitInfo object.
  splitInfo = SplitInfo(splitPoints);
}

template<typename FitnessFunction, typename ObservationType>
size_t HoeffdingNumericSplit<FitnessFunction, ObservationType>::
    MajorityClass() const
{
  // If we haven't yet determined the bins, we must calculate this by hand.
  if (samplesSeen < observationsBeforeBinning)
  {
    arma::Col<size_t> classes(sufficientStatistics.n_rows);
    classes.zeros();

    for (size_t i = 0; i < samplesSeen; ++i)
      classes[labels[i]]++;

    arma::uword majorityClass;
    classes.max(majorityClass);
    return size_t(majorityClass);
  }
  else
  {
    // We've calculated the bins, so we can just sum over the sufficient
    // statistics.
    arma::Col<size_t> classCounts = sum(sufficientStatistics, 1);

    arma::uword maxIndex;
    classCounts.max(maxIndex);
    return size_t(maxIndex);
  }
}

} // namespace tree
} // namespace mlpack

#endif
