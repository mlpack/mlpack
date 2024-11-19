/**
 * @file methods/hoeffding_trees/hoeffding_numeric_split_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of the simple HoeffdingNumericSplit class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_IMPL_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_IMPL_HPP

#include "hoeffding_numeric_split.hpp"

namespace mlpack {

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
    sufficientStatistics(zeros<arma::Mat<size_t>>(numClasses, bins))
{
  observations.zeros();
  labels.zeros();
}

template<typename FitnessFunction, typename ObservationType>
HoeffdingNumericSplit<FitnessFunction, ObservationType>::HoeffdingNumericSplit(
    const size_t numClasses,
    const HoeffdingNumericSplit& other) :
    observations(other.observationsBeforeBinning - 1),
    labels(other.observationsBeforeBinning - 1),
    bins(other.bins),
    observationsBeforeBinning(other.observationsBeforeBinning),
    samplesSeen(0),
    sufficientStatistics(zeros<arma::Mat<size_t>>(numClasses, bins))
{
  observations.zeros();
  labels.zeros();
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
      while (bin < bins - 1 && observations[i] > splitPoints[bin])
        ++bin;

      sufficientStatistics(labels[i], bin)++;
    }
  }

  // If we've gotten to here, then we need to add the point to the sufficient
  // statistics.  What bin does the point fall into?
  size_t bin = 0;
  while (bin < bins - 1 && value > splitPoints[bin])
    ++bin;

  sufficientStatistics(label, bin)++;
}

template<typename FitnessFunction, typename ObservationType>
void HoeffdingNumericSplit<FitnessFunction, ObservationType>::
    EvaluateFitnessFunction(double& bestFitness,
                            double& secondBestFitness) const
{
  secondBestFitness = 0.0; // We can only split one way.
  if (samplesSeen < observationsBeforeBinning)
    bestFitness = 0.0;
  else
    bestFitness = FitnessFunction::Evaluate(sufficientStatistics);
}

template<typename FitnessFunction, typename ObservationType>
void HoeffdingNumericSplit<FitnessFunction, ObservationType>::Split(
    arma::Col<size_t>& childMajorities,
    SplitInfo& splitInfo) const
{
  childMajorities.set_size(sufficientStatistics.n_cols);
  for (size_t i = 0; i < sufficientStatistics.n_cols; ++i)
  {
    arma::uword maxIndex = sufficientStatistics.unsafe_col(i).index_max();
    childMajorities[i] = size_t(maxIndex);
  }

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

    arma::uword majorityClass = classes.index_max();
    return size_t(majorityClass);
  }
  else
  {
    // We've calculated the bins, so we can just sum over the sufficient
    // statistics.
    arma::Col<size_t> classCounts = sum(sufficientStatistics, 1);

    arma::uword maxIndex = classCounts.index_max();
    return size_t(maxIndex);
  }
}

template<typename FitnessFunction, typename ObservationType>
double HoeffdingNumericSplit<FitnessFunction, ObservationType>::
    MajorityProbability() const
{
  // If we haven't yet determined the bins, we must calculate this by hand.
  if (samplesSeen < observationsBeforeBinning)
  {
    arma::Col<size_t> classes(sufficientStatistics.n_rows);
    classes.zeros();

    for (size_t i = 0; i < samplesSeen; ++i)
      classes[labels[i]]++;

    return double(classes.max()) / double(accu(classes));
  }
  else
  {
    // We've calculated the bins, so we can just sum over the sufficient
    // statistics.
    arma::Col<size_t> classCounts = sum(sufficientStatistics, 1);

    return double(classCounts.max()) / double(sum(classCounts));
  }
}

template<typename FitnessFunction, typename ObservationType>
template<typename Archive>
void HoeffdingNumericSplit<FitnessFunction, ObservationType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(samplesSeen));
  ar(CEREAL_NVP(observationsBeforeBinning));
  ar(CEREAL_NVP(bins));

  if (samplesSeen >= observationsBeforeBinning)
  {
    // The binning has happened, so we only need to save the resulting bins.
    ar(CEREAL_NVP(splitPoints));
    ar(CEREAL_NVP(sufficientStatistics));

    if (cereal::is_loading<Archive>())
    {
      // Clean other objects.
      observations.clear();
      labels.clear();
    }
  }
  else
  {
    // The binning has not happened yet, so we only need to save the information
    // required before binning.
    if (cereal::is_loading<Archive>())
    {
      observations.zeros(observationsBeforeBinning);
      labels.zeros(observationsBeforeBinning);
    }

    // Save the number of classes.
    size_t numClasses;
    if (cereal::is_saving<Archive>())
      numClasses = sufficientStatistics.n_rows;
    ar(CEREAL_NVP(numClasses));
    ar(CEREAL_NVP(observations));
    ar(CEREAL_NVP(labels));

    if (cereal::is_loading<Archive>())
    {
      // Clean other objects.
      splitPoints.clear();
      sufficientStatistics.zeros(numClasses, bins);
    }
  }
}

} // namespace mlpack

#endif
