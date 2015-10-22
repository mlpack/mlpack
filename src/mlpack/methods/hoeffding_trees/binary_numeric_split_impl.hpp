/**
 * @file binary_numeric_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the BinaryNumericSplit class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_BINARY_NUMERIC_SPLIT_IMPL_HPP

// In case it hasn't been included yet.
#include "binary_numeric_split.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction, typename ObservationType>
BinaryNumericSplit<FitnessFunction, ObservationType>::BinaryNumericSplit(
    const size_t numClasses) :
    classCounts(numClasses),
    isAccurate(true),
    bestSplit(std::numeric_limits<ObservationType>::min())
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
double BinaryNumericSplit<FitnessFunction, ObservationType>::
    EvaluateFitnessFunction()
{
  // Unfortunately, we have to iterate over the map.
  bestSplit = std::numeric_limits<ObservationType>::min();

  // Initialize the sufficient statistics.
  arma::Mat<size_t> counts(classCounts.n_elem, 2);
  counts.col(0).zeros();
  counts.col(1) = classCounts;

  double bestValue = FitnessFunction::Evaluate(counts);

  for (typename std::multimap<ObservationType, size_t>::const_iterator it =
      sortedElements.begin(); it != sortedElements.end(); ++it)
  {
    // Move the point to the right side of the split.
    --counts((*it).second, 1);
    ++counts((*it).second, 0);

    // TODO: skip ahead if the next value is the same.
    const double value = FitnessFunction::Evaluate(counts);
    if (value > bestValue)
    {
      bestValue = value;
      bestSplit = (*it).first;
    }
  }

  isAccurate = true;
  return bestValue;
}

template<typename FitnessFunction, typename ObservationType>
void BinaryNumericSplit<FitnessFunction, ObservationType>::Split(
    arma::Col<size_t>& childMajorities,
    SplitInfo& splitInfo)
{
  if (!isAccurate)
    EvaluateFitnessFunction();

  // Make one child for each side of the split.
  childMajorities.set_size(2);

  arma::Mat<size_t> counts(classCounts.n_elem, 2);
  counts.col(0).zeros();
  counts.col(1) = classCounts;

  for (typename std::multimap<ObservationType, size_t>::const_iterator it =
      sortedElements.begin(); (*it).first < bestSplit; ++it)
  {
    // Move the point to the correct side of the split.
    --counts((*it).second, 1);
    ++counts((*it).second, 0);
  }

  // Calculate the majority classes of the children.
  arma::uword maxIndex;
  counts.unsafe_col(0).max(maxIndex);
  childMajorities[0] = size_t(maxIndex);
  counts.unsafe_col(1).max(maxIndex);
  childMajorities[1] = size_t(maxIndex);

  // Create the according SplitInfo object.
  arma::vec splitPoints(1);
  splitPoints[0] = double(bestSplit);
  splitInfo = SplitInfo(splitPoints);
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
