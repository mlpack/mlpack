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
                  const data::DatasetInfo& datasetInfo,
                  const double successProbability,
                  const size_t maxSamples,
                  std::unordered_map<size_t, std::pair<size_t, size_t>>*
                      dimensionMappingsIn) :
    dimensionMappings((dimensionMappingsIn != NULL) ? dimensionMappingsIn :
        new std::unordered_map<size_t, std::pair<size_t, size_t>>()),
    ownsMappings(dimensionMappingsIn == NULL),
    numSamples(0),
    numClasses(numClasses),
    maxSamples(maxSamples),
    datasetInfo(datasetInfo),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    categoricalSplit(0),
    numericSplit()
{
  // Do we need to generate the mappings too?
  if (ownsMappings)
  {
    for (size_t i = 0; i < dimensionality; ++i)
    {
      if (datasetInfo.Type(i) == data::Datatype::categorical)
      {
        categoricalSplits.push_back(
            CategoricalSplitType(datasetInfo.NumMappings(i), numClasses));
        (*dimensionMappings)[i] = std::make_pair(data::Datatype::categorical,
            categoricalSplits.size() - 1);
      }
      else
      {
        numericSplits.push_back(NumericSplitType(numClasses));
        (*dimensionMappings)[i] = std::make_pair(data::Datatype::numeric,
            numericSplits.size() - 1);
      }
    }
  }
  else
  {
    for (size_t i = 0; i < dimensionality; ++i)
    {
      if (datasetInfo.Type(i) == data::Datatype::categorical)
      {
        categoricalSplits.push_back(
            CategoricalSplitType(datasetInfo.NumMappings(i), numClasses));
      }
      else
      {
        numericSplits.push_back(NumericSplitType(numClasses));
      }
    }
  }
}

template<typename FitnessFunction,
         typename NumericSplitType,
         typename CategoricalSplitType>
HoeffdingSplit<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    ~HoeffdingSplit()
{
  if (ownsMappings)
    delete dimensionMappings;
}

template<typename FitnessFunction,
         typename NumericSplitType,
         typename CategoricalSplitType>
template<typename VecType>
void HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const VecType& point, const size_t label)
{
  if (splitDimension == size_t(-1))
  {
    ++numSamples;
    size_t numericIndex = 0;
    size_t categoricalIndex = 0;
    for (size_t i = 0; i < point.n_rows; ++i)
    {
      if (datasetInfo.Type(i) == data::Datatype::categorical)
        categoricalSplits[categoricalIndex++].Train(point[i], label);
      else if (datasetInfo.Type(i) == data::Datatype::numeric)
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
>::SplitCheck()
{
  // Do nothing if we've already split.
  if (splitDimension != size_t(-1))
    return 0;

  // Check the fitness of each dimension.  Then we'll use a Hoeffding bound
  // somehow.

  // Calculate epsilon, the value we need things to be greater than.
  const double rSquared = std::pow(FitnessFunction::Range(numClasses), 2.0);
  const double epsilon = std::sqrt(rSquared *
      std::log(1.0 / (1.0 - successProbability)) / (2 * numSamples));

  arma::vec gains(categoricalSplits.size() + numericSplits.size());
  for (size_t i = 0; i < gains.n_elem; ++i)
  {
    size_t type = dimensionMappings->at(i).first;
    size_t index = dimensionMappings->at(i).second;
    if (type == data::Datatype::categorical)
      gains[i] = categoricalSplits[index].EvaluateFitnessFunction();
    else if (type == data::Datatype::numeric)
      gains[i] = numericSplits[index].EvaluateFitnessFunction();
  }

  // Now find the largest and second-largest.
  double largest = -DBL_MAX;
  size_t largestIndex = 0;
  double secondLargest = -DBL_MAX;
  for (size_t i = 0; i < gains.n_elem; ++i)
  {
    if (gains[i] > largest)
    {
      secondLargest = largest;
      largest = gains[i];
      largestIndex = i;
    }
    else if (gains[i] > secondLargest)
    {
      secondLargest = gains[i];
    }
  }

  // Are these far enough apart to split?
  if (largest - secondLargest > epsilon || numSamples > maxSamples)
  {
    // Split!
    splitDimension = largestIndex;
    if (datasetInfo.Type(largestIndex) == data::Datatype::categorical)
    {
      // I don't know if this should be here.
      majorityClass = categoricalSplits[largestIndex].MajorityClass();
      return datasetInfo.NumMappings(largestIndex);
    }
    else
    {
      majorityClass = numericSplits[largestIndex].MajorityClass();
      return numericSplits[largestIndex].Bins();
    }
  }
  else
  {
    return 0; // Don't split.
  }
}

template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
size_t HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::MajorityClass() const
{
  // If the node is not split yet, we have to grab the majority class from any
  // of the structures figuring out what to split on.
  if (splitDimension == size_t(-1))
  {
    // Grab majority class from splits.
    if (categoricalSplits.size() > 0)
      majorityClass = categoricalSplits[0].MajorityClass();
    else
      majorityClass = numericSplits[0].MajorityClass();
  }

  return majorityClass;
}

template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
size_t& HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::MajorityClass()
{
  return majorityClass;
}

template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
template<typename VecType>
size_t HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CalculateDirection(const VecType& point) const
{
  // Don't call this before the node is split...
  if (datasetInfo.Type(splitDimension) == data::Datatype::numeric)
    return numericSplit.CalculateDirection(point[splitDimension]);
  else if (datasetInfo.Type(splitDimension) == data::Datatype::categorical)
    return categoricalSplit.CalculateDirection(point[splitDimension]);
  else
    return 0; // Not sure what to do here...
}

template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
template<typename VecType>
size_t HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& /* point */) const
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
template<typename StreamingDecisionTreeType>
void HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CreateChildren(std::vector<StreamingDecisionTreeType>& children)
{
  // Create the children.
  arma::Col<size_t> childMajorities;
  if (dimensionMappings->at(splitDimension).first ==
      data::Datatype::categorical)
  {
    categoricalSplits[dimensionMappings->at(splitDimension).second].Split(
        childMajorities, categoricalSplit);
  }
  else if (dimensionMappings->at(splitDimension).first ==
           data::Datatype::numeric)
  {
    numericSplits[dimensionMappings->at(splitDimension).second].Split(
        childMajorities, numericSplit);
  }

  // We already know what the splitDimension will be.
  const size_t dimensionality = numericSplits.size() + categoricalSplits.size();
  for (size_t i = 0; i < childMajorities.n_elem; ++i)
  {
    children.push_back(StreamingDecisionTreeType(datasetInfo, dimensionality,
        numClasses, successProbability, numSamples, dimensionMappings));
    children[i].MajorityClass() = childMajorities[i];
  }

  // Eliminate now-unnecessary split information.
  numericSplits.clear();
  categoricalSplits.clear();
}

} // namespace tree
} // namespace mlpack

#endif
