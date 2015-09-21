/**
 * @file hoeffding_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the HoeffdingSplit class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_SPLIT_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_SPLIT_IMPL_HPP

namespace mlpack {
namespace tree {

template<typename FitnessFunction,
         typename NumericSplitType,
         typename CategoricalSplitType>
HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingSplit(const size_t dimensionality,
                  const size_t numClasses,
                  const DatasetInfo& datasetInfo)
{
  for (size_t i = 0; i < dimensionality; ++i)
  {
    if (datasetInfo.Type(i) == Datatype.categorical)
      categoricalSplits.push_back(
          CategoricalSplitType(datasetInfo.NumMappings(), numClasses));
    // else, numeric splits (not yet!)
  }
}

template<typename VecType>
template<typename FitnessFunction,
         typename NumericSplitType,
         typename CategoricalSplitType>
void HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(VecType& point, const size_t label)
{
  if (splitDimension == size_t(-1))
  {
    ++numSamples;
    size_t numericIndex = 0;
    size_t categoricalIndex = 0;
    for (size_t i = 0; i < point.n_rows; ++i)
    {
      if (datasetInfo.Type(i) == Datatype.categorical)
        categoricalSplits[categoricalIndex++].Train(point[i], label);
      else if (datasetInfo.Type(i) == Datatype.numeric)
        numericSplits[numericIndex++].Train(point[i], label);
    }
  }
  else
  {
    // Already split.
    // But we should probably pass it down anyway.
  }
}

template<typename FitnessFunction,
         typename NumericSplitType,
         typename CategoricalSplitType>
size_t HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SplitCheck() const
{
  // Do nothing if we've already split.
  if (splitDimension == size_t(-1))
    return 0;

  // Check the fitness of each dimension.  Then we'll use a Hoeffding bound
  // somehow.

  // Calculate epsilon, the value we need things to be greater than.
  const double rSquared = std::pow(FitnessFunction::Range(numClasses), 2.0);
  const double epsilon = std::sqrt(rSquared *
      std::log(1.0 / successProbability) / (2 * numSamples));

  arma::vec gains(categoricalSplits.size() + numericSplits.size());
  for (size_t i = 0; i < categoricalSplits.size(); ++i)
    gains[i] = categoricalSplits[i].EvaluateFitnessFunction();

  // Now find the largest and second-largest.
  double largest = -DBL_MAX;
  size_t largestIndex = 0;
  double secondLargest = -DBL_MAX;
  size_t secondLargestIndex = 0;
  for (size_t i = 0; i < gains.n_elem; ++i)
  {
    if (gains[i] > largest)
    {
      secondLargest = largest;
      secondLargestIndex = largestIndex;
      largest = gains[i];
      largestIndex = i;
    }
    else if (gains[i] > secondLargest)
    {
      secondLargest = gains[i];
      secondLargestIndex = i;
    }
  }

  // Are these far enough apart to split?
  if (largest - secondLargest > epsilon)
  {
    // Split!
    splitDimension = largestIndex;
    if (datasetInfo[largestIndex].Type == Datatype.categorical)
    {
      // I don't know if this should be here.
      majorityClass = categoricalSplit[largestIndex].MajorityClass();
      return datasetInfo[largestIndex].NumMappings();
    }
    else
    {
      majorityClass = 0;
      return 0; // I have no idea what to do yet.
    }
  }
}

template<typename VecType>
template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
size_t HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CalculateDirection(VecType& point) const
{
  // Don't call this before the node is split...
  if (datasetInfo.Type(splitDimension) == Datatype::numeric)
    return numericSplit.CalculateDirection(point[splitDimension]);
  else if (datasetInfo.Type(splitDimension) == Datatype::categorical)
    return categoricalSplit.CalculateDirection(point[splitDimension]);
}

template<typename VecType>
template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
void HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& point) const
{
  // We're a leaf (or being considered a leaf), so classify based on what we
  // know.
  return majorityClass;
}

template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
void HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CreateChildren(std::vector<StreamingDecisionTreeType>& children)
{
  // We already know what the splitDimension will be.
  size_t numericSplitIndex = 0;
  size_t categoricalSplitIndex = 0;
  for (size_t i = 0; i < splitDimension; ++i)
  {
    if (datasetInfo.Type(i) == Datatype::numeric)
      ++numericSplitIndex;
    if (datasetInfo.Type(i) == Datatype::categorical)
      ++categoricalSplitIndex;
  }

  if (datasetInfo.Type(splitDimension) == Datatype::numeric)
  {
    numericSplits[numericSplitIndex + 1].CreateChildren(children, numericSplit);
  }
  else if (datasetInfo.Type(splitDimension) == Datatype::categorical)
  {
    categoricalSplits[categoricalSplitIndex + 1].CreateChildren(children,
        categoricalSplit);
  }
}

} // namespace tree
} // namespace mlpack

#endif
