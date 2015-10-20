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
    datasetInfo(const_cast<data::DatasetInfo*>(&datasetInfo)),
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
      if (datasetInfo->Type(i) == data::Datatype::categorical)
        categoricalSplits[categoricalIndex++].Train(point[i], label);
      else if (datasetInfo->Type(i) == data::Datatype::numeric)
        numericSplits[numericIndex++].Train(point[i], label);
    }

    // Grab majority class from splits.
    if (categoricalSplits.size() > 0)
    {
      majorityClass = categoricalSplits[0].MajorityClass();
      majorityProbability = categoricalSplits[0].MajorityProbability();
    }
    else
    {
      majorityClass = numericSplits[0].MajorityClass();
      majorityProbability = numericSplits[0].MajorityProbability();
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
    const size_t type = dimensionMappings->at(largestIndex).first;
    const size_t index = dimensionMappings->at(largestIndex).second;
    if (type == data::Datatype::categorical)
    {
      // I don't know if this should be here.
      majorityClass = categoricalSplits[index].MajorityClass();
      return categoricalSplits[index].NumChildren();
    }
    else
    {
      majorityClass = numericSplits[index].MajorityClass();
      return numericSplits[index].NumChildren();
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
  if (datasetInfo->Type(splitDimension) == data::Datatype::numeric)
    return numericSplit.CalculateDirection(point[splitDimension]);
  else if (datasetInfo->Type(splitDimension) == data::Datatype::categorical)
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
template<typename VecType>
void HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& /* point */,
            size_t& prediction,
            double& probability) const
{
  prediction = majorityClass;
  probability = majorityProbability;
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
    children.push_back(StreamingDecisionTreeType(*datasetInfo, dimensionality,
        numClasses, successProbability, maxSamples, dimensionMappings));
    children[i].MajorityClass() = childMajorities[i];
  }

  // Eliminate now-unnecessary split information.
  numericSplits.clear();
  categoricalSplits.clear();
}

template<
    typename FitnessFunction,
    typename NumericSplitType,
    typename CategoricalSplitType
>
template<typename Archive>
void HoeffdingSplit<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(splitDimension, "splitDimension");
  ar & CreateNVP(dimensionMappings, "dimensionMappings");
  ar & CreateNVP(ownsMappings, "ownsMappings");
  ar & CreateNVP(datasetInfo, "datasetInfo");
  ar & CreateNVP(majorityClass, "majorityClass");
  ar & CreateNVP(majorityProbability, "majorityProbability");

  // Depending on whether or not we have split yet, we may need to save
  // different things.
  if (splitDimension == size_t(-1))
  {
    // We have not yet split.  So we have to serialize the splits.
    ar & CreateNVP(numSamples, "numSamples");
    ar & CreateNVP(numClasses, "numClasses");
    ar & CreateNVP(maxSamples, "maxSamples");
    ar & CreateNVP(successProbability, "successProbability");

    // Serialize the splits, but not if we haven't seen any samples yet (in
    // which case we can just reinitialize).
    if (Archive::is_loading::value)
    {
      // Re-initialize all of the splits.
      numericSplits.clear();
      categoricalSplits.clear();
      for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
      {
        if (datasetInfo->Type(i) == data::Datatype::categorical)
          categoricalSplits.push_back(CategoricalSplitType(
              datasetInfo->NumMappings(i), numClasses));
        else
          numericSplits.push_back(NumericSplitType(numClasses));
      }

      // Clear things we don't need.
      categoricalSplit = typename CategoricalSplitType::SplitInfo(numClasses);
      numericSplit = typename NumericSplitType::SplitInfo();
    }

    // There's no need to serialize if there's no information contained in the
    // splits.
    if (numSamples == 0)
      return;

    // Serialize numeric splits.
    for (size_t i = 0; i < numericSplits.size(); ++i)
    {
      std::ostringstream name;
      name << "numericSplit" << i;
      ar & CreateNVP(numericSplits[i], name.str());
    }

    // Serialize categorical splits.
    for (size_t i = 0; i < categoricalSplits.size(); ++i)
    {
      std::ostringstream name;
      name << "categoricalSplit" << i;
      ar & CreateNVP(categoricalSplits[i], name.str());
    }
  }
  else
  {
    // We have split, so we only need to save the split.
    if (datasetInfo->Type(splitDimension) == data::Datatype::categorical)
      ar & CreateNVP(categoricalSplit, "categoricalSplit");
    else
      ar & CreateNVP(numericSplit, "numericSplit");

    if (Archive::is_loading::value)
    {
      numericSplits.clear();
      categoricalSplits.clear();

      numSamples = 0;
      numClasses = 0;
      maxSamples = 0;
      successProbability = 0.0;
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif
