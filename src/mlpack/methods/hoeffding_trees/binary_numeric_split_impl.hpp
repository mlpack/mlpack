/**
 * @file binary_numeric_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the BinaryNumericSplit class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_IMPL_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_IMPL_HPP

// In case it hasn't been included yet.
#include "binary_numeric_split.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction, typename ObservationType>
BinaryNumericSplit<FitnessFunction, ObservationType>::BinaryNumericSplit(
    const size_t numClasses) :
    classCounts(numClasses),
    bestSplit(std::numeric_limits<ObservationType>::min()),
    isAccurate(true)
{
  // Zero out class counts.
  classCounts.zeros();
}

template<typename FitnessFunction, typename ObservationType>
BinaryNumericSplit<FitnessFunction, ObservationType>::BinaryNumericSplit(
    const size_t numClasses,
    const BinaryNumericSplit& /* other */) :
    classCounts(numClasses),
    bestSplit(std::numeric_limits<ObservationType>::min()),
    isAccurate(true)
{
  // Zero out class counts.
  classCounts.zeros();
}

template<typename FitnessFunction, typename ObservationType>
void BinaryNumericSplit<FitnessFunction, ObservationType>::Train(
    ObservationType value,
    const size_t label)
{
  // Push it into the multimap, and update the class counts.
  sortedElements.insert(std::pair<ObservationType, size_t>(value, label));
  ++classCounts[label];

  // Whatever we have cached is no longer valid.
  isAccurate = false;
}

template<typename FitnessFunction, typename ObservationType>
void BinaryNumericSplit<FitnessFunction, ObservationType>::
    EvaluateFitnessFunction(double& bestFitness,
                            double& secondBestFitness)
{
  // Unfortunately, we have to iterate over the map.
  bestSplit = std::numeric_limits<ObservationType>::min();

  // Initialize the sufficient statistics.
  arma::Mat<size_t> counts(classCounts.n_elem, 2);
  counts.col(0).zeros();
  counts.col(1) = classCounts;

  bestFitness = FitnessFunction::Evaluate(counts);
  secondBestFitness = 0.0;

  // Initialize to the first observation, so we don't calculate gain on the
  // first iteration (it will be 0).
  ObservationType lastObservation = (*sortedElements.begin()).first;
  size_t lastClass = classCounts.n_elem;
  for (typename std::multimap<ObservationType, size_t>::const_iterator it =
      sortedElements.begin(); it != sortedElements.end(); ++it)
  {
    // If this value is the same as the last, or if this is the first value, or
    // we have the same class as the previous observation, don't calculate the
    // gain---it can't be any better.  (See Fayyad and Irani, 1991.)
    if (((*it).first != lastObservation) || ((*it).second != lastClass))
    {
      lastObservation = (*it).first;
      lastClass = (*it).second;

      const double value = FitnessFunction::Evaluate(counts);
      if (value > bestFitness)
      {
        bestFitness = value;
        bestSplit = (*it).first;
      }
      else if (value > secondBestFitness)
      {
        secondBestFitness = value;
      }
    }

    // Move the point to the right side of the split.
    --counts((*it).second, 1);
    ++counts((*it).second, 0);
  }

  isAccurate = true;
}

template<typename FitnessFunction, typename ObservationType>
void BinaryNumericSplit<FitnessFunction, ObservationType>::Split(
    arma::Col<size_t>& childMajorities,
    SplitInfo& splitInfo)
{
  if (!isAccurate)
  {
    double bestGain, secondBestGain;
    EvaluateFitnessFunction(bestGain, secondBestGain);
  }

  // Make one child for each side of the split.
  childMajorities.set_size(2);

  arma::Mat<size_t> counts(classCounts.n_elem, 2);
  counts.col(0).zeros();
  counts.col(1) = classCounts;

  double min = DBL_MAX;
  double max = -DBL_MAX;
  for (typename std::multimap<ObservationType, size_t>::const_iterator it =
      sortedElements.begin();// (*it).first < bestSplit; ++it)
      it != sortedElements.end(); ++it)
  {
    // Move the point to the correct side of the split.
    if ((*it).first < bestSplit)
    {
      --counts((*it).second, 1);
      ++counts((*it).second, 0);
    }
    if ((*it).first < min)
      min = (*it).first;
    if ((*it).first > max)
      max = (*it).first;
  }

  // Calculate the majority classes of the children.
  arma::uword maxIndex;
  counts.unsafe_col(0).max(maxIndex);
  childMajorities[0] = size_t(maxIndex);
  counts.unsafe_col(1).max(maxIndex);
  childMajorities[1] = size_t(maxIndex);

  // Create the according SplitInfo object.
  splitInfo = SplitInfo(bestSplit);
}

template<typename FitnessFunction, typename ObservationType>
size_t BinaryNumericSplit<FitnessFunction, ObservationType>::MajorityClass()
    const
{
  arma::uword maxIndex;
  classCounts.max(maxIndex);
  return size_t(maxIndex);
}

template<typename FitnessFunction, typename ObservationType>
double BinaryNumericSplit<FitnessFunction, ObservationType>::
    MajorityProbability() const
{
  return double(arma::max(classCounts)) / double(arma::accu(classCounts));
}

template<typename FitnessFunction, typename ObservationType>
template<typename Archive>
void BinaryNumericSplit<FitnessFunction, ObservationType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  // Serialize.
  ar & data::CreateNVP(sortedElements, "sortedElements");
  ar & data::CreateNVP(classCounts, "classCounts");
}


} // namespace tree
} // namespace mlpack

#endif
