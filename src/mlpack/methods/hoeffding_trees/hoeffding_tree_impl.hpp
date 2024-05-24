/**
 * @file methods/hoeffding_trees/hoeffding_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the HoeffdingTree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_IMPL_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_IMPL_HPP

// In case it hasn't been included yet.
#include "hoeffding_tree.hpp"

namespace mlpack {

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingTree() :
    dimensionMappings(
        new std::unordered_map<size_t, std::pair<size_t, size_t>>()),
    ownsMappings(true),
    numSamples(0),
    numClasses(0),
    maxSamples(size_t(-1)),
    checkInterval(100),
    minSamples(100),
    datasetInfo(new data::DatasetInfo()),
    ownsInfo(true),
    successProbability(0.95),
    splitDimension(size_t(-1)),
    majorityClass(0),
    majorityProbability(0.0),
    categoricalSplit(0),
    numericSplit()
{
  // Nothing to do.
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingTree(const size_t dimensionality,
                 const size_t numClasses,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval,
                 const size_t minSamples,
                 const CategoricalSplitType<FitnessFunction>&
                     categoricalSplitIn,
                 const NumericSplitType<FitnessFunction>& numericSplitIn,
                 std::unordered_map<size_t, std::pair<size_t, size_t>>*
                     dimensionMappingsIn) :
    dimensionMappings((dimensionMappingsIn != NULL) ? dimensionMappingsIn :
        new std::unordered_map<size_t, std::pair<size_t, size_t>>()),
    ownsMappings(dimensionMappingsIn == NULL),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    minSamples(minSamples),
    datasetInfo(new data::DatasetInfo(dimensionality)),
    ownsInfo(true),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    majorityClass(0),
    majorityProbability(0.0),
    categoricalSplit(0),
    numericSplit()
{
  // Do we need to generate the mappings too?
  if (ownsMappings)
  {
    ResetTree(categoricalSplitIn, numericSplitIn);
  }
  else
  {
    // All dimensions are numeric.
    for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
    {
      numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses,
          numericSplitIn));
    }
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingTree(const data::DatasetInfo& datasetInfo,
                 const size_t numClasses,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval,
                 const size_t minSamples,
                 const CategoricalSplitType<FitnessFunction>&
                     categoricalSplitIn,
                 const NumericSplitType<FitnessFunction>& numericSplitIn,
                 std::unordered_map<size_t, std::pair<size_t, size_t>>*
                     dimensionMappingsIn,
                 const bool copyDatasetInfo) :
    dimensionMappings((dimensionMappingsIn != NULL) ? dimensionMappingsIn :
        new std::unordered_map<size_t, std::pair<size_t, size_t>>()),
    ownsMappings(dimensionMappingsIn == NULL),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    minSamples(minSamples),
    datasetInfo(copyDatasetInfo ? new data::DatasetInfo(datasetInfo) :
        &datasetInfo),
    ownsInfo(copyDatasetInfo),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    majorityClass(0),
    majorityProbability(0.0),
    categoricalSplit(0),
    numericSplit()
{
  // Do we need to generate the mappings too?
  if (ownsMappings)
  {
    ResetTree(categoricalSplitIn, numericSplitIn);
  }
  else
  {
    for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
    {
      if (datasetInfo.Type(i) == data::Datatype::categorical)
      {
        categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
            datasetInfo.NumMappings(i), numClasses, categoricalSplitIn));
      }
      else
      {
        numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses,
            numericSplitIn));
      }
    }
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingTree(const MatType& data,
                 const arma::Row<size_t>& labels,
                 const size_t numClasses,
                 const bool batchTraining,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval,
                 const size_t minSamples,
                 const CategoricalSplitType<FitnessFunction>&
                     categoricalSplitIn,
                 const NumericSplitType<FitnessFunction>& numericSplitIn) :
    dimensionMappings(NULL),
    ownsMappings(false),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    minSamples(minSamples),
    datasetInfo(new data::DatasetInfo(data.n_rows)),
    ownsInfo(true),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    majorityClass(0),
    majorityProbability(0.0),
    categoricalSplit(0),
    numericSplit()
{
  // Reset the tree.
  ResetTree(categoricalSplitIn, numericSplitIn);

  // Now train.
  Train(data, labels, numClasses, batchTraining);
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingTree(const MatType& data,
                 const data::DatasetInfo& datasetInfoIn,
                 const arma::Row<size_t>& labels,
                 const size_t numClasses,
                 const bool batchTraining,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval,
                 const size_t minSamples,
                 const CategoricalSplitType<FitnessFunction>&
                     categoricalSplitIn,
                 const NumericSplitType<FitnessFunction>& numericSplitIn) :
    dimensionMappings(NULL),
    ownsMappings(false),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    minSamples(minSamples),
    datasetInfo(new data::DatasetInfo(datasetInfoIn)),
    ownsInfo(true),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    majorityClass(0),
    majorityProbability(0.0),
    categoricalSplit(0),
    numericSplit()
{
  // Reset the tree.
  ResetTree(categoricalSplitIn, numericSplitIn);

  // Now train.
  Train(data, labels, numClasses, batchTraining);
}

// Copy constructor.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    HoeffdingTree(const HoeffdingTree& other) :
    numericSplits(other.numericSplits),
    categoricalSplits(other.categoricalSplits),
    dimensionMappings(new std::unordered_map<size_t,
        std::pair<size_t, size_t>>(*other.dimensionMappings)),
    ownsMappings(true),
    numSamples(other.numSamples),
    numClasses(other.numClasses),
    maxSamples(other.maxSamples),
    checkInterval(other.checkInterval),
    minSamples(other.minSamples),
    datasetInfo(new data::DatasetInfo(*other.datasetInfo)),
    ownsInfo(true),
    successProbability(other.successProbability),
    splitDimension(other.splitDimension),
    majorityClass(other.majorityClass),
    majorityProbability(other.majorityProbability),
    categoricalSplit(other.categoricalSplit),
    numericSplit(other.numericSplit)
{
  // Copy each of the children.
  for (size_t i = 0; i < other.children.size(); ++i)
  {
    children.push_back(new HoeffdingTree(*other.children[i]));

    // Delete copied datasetInfo and dimension mappings.
    delete children[i]->datasetInfo;
    children[i]->datasetInfo = this->datasetInfo;
    children[i]->ownsInfo = false;

    delete children[i]->dimensionMappings;
    children[i]->dimensionMappings = this->dimensionMappings;
    children[i]->ownsMappings = false;
  }
}

// Move constructor.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    HoeffdingTree(HoeffdingTree&& other) :
    numericSplits(std::move(other.numericSplits)),
    categoricalSplits(std::move(other.categoricalSplits)),
    dimensionMappings(other.dimensionMappings),
    ownsMappings(true),
    numSamples(other.numSamples),
    numClasses(other.numClasses),
    maxSamples(other.maxSamples),
    checkInterval(other.checkInterval),
    minSamples(other.minSamples),
    datasetInfo(other.datasetInfo),
    ownsInfo(true),
    successProbability(other.successProbability),
    splitDimension(other.splitDimension),
    majorityClass(other.majorityClass),
    majorityProbability(other.majorityProbability),
    categoricalSplit(std::move(other.categoricalSplit)),
    numericSplit(std::move(other.numericSplit))
{
  // Remove pointers.
  other.dimensionMappings = nullptr;
  other.datasetInfo = nullptr;

  // Reset primary type variables.
  other.numSamples = 0;
  other.numClasses = 0;
  other.checkInterval = 0;
  other.minSamples = 0;
  other.successProbability = 0.0;
  other.splitDimension = 0;
  other.majorityClass = 0;
  other.majorityProbability = 0.0;
}

// Copy assignment operator.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>&
    HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    operator=(const HoeffdingTree& other)
{
  if (this != &other)
  {
    numericSplits = other.numericSplits;
    categoricalSplits = other.categoricalSplits;
    dimensionMappings = new std::unordered_map<size_t,
        std::pair<size_t, size_t>>(*other.dimensionMappings);
    ownsMappings = true;
    numSamples = other.numSamples;
    numClasses = other.numClasses;
    maxSamples = other.maxSamples;
    checkInterval = other.checkInterval;
    minSamples = other.minSamples;
    datasetInfo = new data::DatasetInfo(*other.datasetInfo);
    ownsInfo = true;
    successProbability = other.successProbability;
    splitDimension = other.splitDimension;
    majorityClass = other.majorityClass;
    majorityProbability = other.majorityProbability;
    categoricalSplit = other.categoricalSplit;
    numericSplit = other.numericSplit;

    // Copy each of the children.
    for (size_t i = 0; i < other.children.size(); ++i)
    {
      children.push_back(new HoeffdingTree(*other.children[i]));

      // Delete copied datasetInfo and dimension mappings.
      delete children[i]->datasetInfo;
      children[i]->datasetInfo = this->datasetInfo;
      children[i]->ownsInfo = false;

      delete children[i]->dimensionMappings;
      children[i]->dimensionMappings = this->dimensionMappings;
      children[i]->ownsMappings = false;
    }
  }
  return *this;
}

// Move assignment operator.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>&
    HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    operator=(HoeffdingTree&& other)
{
  if (this != &other)
  {
    numericSplits = std::move(other.numericSplits);
    categoricalSplits = std::move(other.categoricalSplits);
    dimensionMappings = other.dimensionMappings;
    ownsMappings = true;
    numSamples = other.numSamples;
    numClasses = other.numClasses;
    maxSamples = other.maxSamples;
    checkInterval = other.checkInterval;
    minSamples = other.minSamples;
    datasetInfo = other.datasetInfo;
    ownsInfo = true;
    successProbability = other.successProbability;
    splitDimension = other.splitDimension;
    majorityClass = other.majorityClass;
    majorityProbability = other.majorityProbability;
    categoricalSplit = std::move(other.categoricalSplit);
    numericSplit = std::move(other.numericSplit);

    // Remove pointers.
    other.dimensionMappings = nullptr;
    other.datasetInfo = nullptr;

    // Reset primary type variables.
    other.numSamples = 0;
    other.numClasses = 0;
    other.checkInterval = 0;
    other.minSamples = 0;
    other.successProbability = 0.0;
    other.splitDimension = 0;
    other.majorityClass = 0;
    other.majorityProbability = 0.0;
  }
  return *this;
}


template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    ~HoeffdingTree()
{
  if (ownsMappings)
    delete dimensionMappings;
  if (ownsInfo)
    delete datasetInfo;
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& data,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const bool batchTraining,
         const std::optional<double> successProbability,
         const std::optional<size_t> maxSamples,
         const std::optional<size_t> checkInterval)
{
  Train(data, labels, numClasses, batchTraining,
      (successProbability.has_value()) ? successProbability.value() :
          this->successProbability,
      (maxSamples.has_value()) ? maxSamples.value() : this->maxSamples,
      (checkInterval.has_value()) ? checkInterval.value() : this->checkInterval,
          this->minSamples);
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& data,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const bool batchTraining,
         const double successProbability,
         const size_t maxSamples,
         const size_t checkInterval,
         const size_t minSamples)
{
  this->successProbability = successProbability;
  this->maxSamples = maxSamples;
  this->checkInterval = checkInterval;
  this->minSamples = minSamples;

  // We need to reset the tree either if the user asked for it, or if they
  // passed data whose dimensionality is different than our datasetInfo object.
  if (data.n_rows != datasetInfo->Dimensionality() ||
      (numClasses != 0 && numClasses != this->numClasses))
  {
    // Create a new datasetInfo, which assumes that all features are numeric.
    if (ownsInfo)
      delete datasetInfo;
    datasetInfo = new data::DatasetInfo(data.n_rows);
    ownsInfo = true;

    // Set the number of classes correctly.
    if (numClasses != 0)
      this->numClasses = numClasses;

    if (this->numClasses == 0)
    {
      throw std::invalid_argument("HoeffdingTree::Train(): must specify number "
          "of classes!");
    }

    ResetTree();
  }

  TrainInternal(data, labels, batchTraining);
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& data,
         const data::DatasetInfo& info,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const bool batchTraining,
         const std::optional<double> successProbability,
         const std::optional<size_t> maxSamples,
         const std::optional<size_t> checkInterval)
{
  Train(data, info, labels, numClasses, batchTraining,
      (successProbability.has_value()) ? successProbability.value() :
          this->successProbability,
      (maxSamples.has_value()) ? maxSamples.value() : this->maxSamples,
      (checkInterval.has_value()) ? checkInterval.value() : this->checkInterval,
          this->minSamples);
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& data,
         const data::DatasetInfo& info,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const bool batchTraining,
         const double successProbability,
         const size_t maxSamples,
         const size_t checkInterval,
         const size_t minSamples)
{
  this->successProbability = successProbability;
  this->maxSamples = maxSamples;
  this->checkInterval = checkInterval;
  this->minSamples = minSamples;

  // We need to reset the tree either if the user asked for it, or if they
  // passed data whose dimensionality is different than our datasetInfo object.
  if (data.n_rows != datasetInfo->Dimensionality() ||
      (numClasses != 0 && numClasses != this->numClasses))
  {
    // Set the number of classes correctly.
    if (numClasses != 0)
      this->numClasses = numClasses;

    if (this->numClasses == 0)
    {
      throw std::invalid_argument("HoeffdingTree::Train(): must specify number "
          "of classes!");
    }

    Reset(info, this->numClasses);
  }
  else if (datasetInfo != &info)
  {
    // Take over new DatasetInfo.
    if (ownsInfo)
      delete datasetInfo;
    datasetInfo = &info;
    ownsInfo = false;
  }

  TrainInternal(data, labels, batchTraining);
}

//! Train on one point.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename VecType>
void HoeffdingTree<
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

    // Check for a split, if we should.
    if (numSamples % checkInterval == 0)
    {
      const size_t numChildren = SplitCheck();
      if (numChildren > 0)
      {
        // We need to add a bunch of children.
        // Delete children, if we have them.
        children.clear();
        CreateChildren();
      }
    }
  }
  else
  {
    // Already split.  Pass the training point to the relevant child.
    size_t direction = CalculateDirection(point);
    children[direction]->Train(point, label);
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SplitCheck()
{
  // Do nothing if we've already split.
  if (splitDimension != size_t(-1))
    return 0;

  // If not enough points have been seen, we cannot split.
  if (numSamples <= minSamples)
    return 0;

  // Check the fitness of each dimension.  Then we'll use a Hoeffding bound
  // somehow.

  // Calculate epsilon, the value we need things to be greater than.
  const double rSquared = std::pow(FitnessFunction::Range(numClasses), 2.0);
  const double epsilon = std::sqrt(rSquared *
      std::log(1.0 / (1.0 - successProbability)) / (2 * numSamples));

  // Find the best and second best possible splits.
  double largest = -DBL_MAX;
  size_t largestIndex = 0;
  double secondLargest = -DBL_MAX;
  for (size_t i = 0; i < categoricalSplits.size() + numericSplits.size(); ++i)
  {
    size_t type = dimensionMappings->at(i).first;
    size_t index = dimensionMappings->at(i).second;

    // Some split procedures can split multiple ways, but we only care about the
    // best two splits that can be done in every network.
    double bestGain = 0.0;
    double secondBestGain = 0.0;
    if (type == data::Datatype::categorical)
      categoricalSplits[index].EvaluateFitnessFunction(bestGain,
          secondBestGain);
    else if (type == data::Datatype::numeric)
      numericSplits[index].EvaluateFitnessFunction(bestGain, secondBestGain);

    // See if these gains are better than the previous.
    if (bestGain > largest)
    {
      secondLargest = largest;
      largest = bestGain;
      largestIndex = i;
    }
    else if (bestGain > secondLargest)
    {
      secondLargest = bestGain;
    }

    if (secondBestGain > secondLargest)
    {
      secondLargest = secondBestGain;
    }
  }

  // Are these far enough apart to split?
  if ((largest > 0.0) &&
      ((largest - secondLargest > epsilon) || (numSamples > maxSamples) ||
       (epsilon <= 0.05)))
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
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SuccessProbability(const double successProbability)
{
  this->successProbability = successProbability;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->SuccessProbability(successProbability);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::MinSamples(const size_t minSamples)
{
  this->minSamples = minSamples;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->MinSamples(minSamples);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::MaxSamples(const size_t maxSamples)
{
  this->maxSamples = maxSamples;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->MaxSamples(maxSamples);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CheckInterval(const size_t checkInterval)
{
  this->checkInterval = checkInterval;
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->CheckInterval(checkInterval);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
size_t HoeffdingTree<
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

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::NumDescendants() const
{
  size_t nodes = 0;
  std::stack<const HoeffdingTree*> stack;
  stack.push(this); // Push the current tree.
  while (!stack.empty())
  {
    const HoeffdingTree* node = stack.top();
    stack.pop();
    nodes += node->NumChildren();
    for (size_t i = 0; i < node->NumChildren(); ++i)
      stack.push(&node->Child(i));
  }
  return nodes;
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& point) const
{
  if (children.size() == 0)
  {
    // If we're a leaf (or being considered a leaf), classify based on what we
    // know.
    return majorityClass;
  }
  else
  {
    // Otherwise, pass to the right child and let them classify.
    return children[CalculateDirection(point)]->Classify(point);
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& point,
            size_t& prediction,
            double& probability) const
{
  if (children.size() == 0)
  {
    // We are a leaf, so classify accordingly.
    prediction = majorityClass;
    probability = majorityProbability;
  }
  else
  {
    // Pass to the right child and let them do the classification.
    children[CalculateDirection(point)]->Classify(point, prediction,
        probability);
  }
}

//! Batch classification.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const MatType& data, arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Classify(data.col(i));
}

//! Batch classification with probabilities.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const MatType& data,
            arma::Row<size_t>& predictions,
            arma::rowvec& probabilities) const
{
  predictions.set_size(data.n_cols);
  probabilities.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    Classify(data.col(i), predictions[i], probabilities[i]);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CreateChildren()
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
  for (size_t i = 0; i < childMajorities.n_elem; ++i)
  {
    // We need to also give our split objects to the new children, so that
    // parameters for the splits can be passed down.  But if we have no
    // categorical or numeric features, we can't pass anything but the
    // defaults...
    if (categoricalSplits.size() == 0)
    {
      // Pass a default categorical split.
      children.push_back(new HoeffdingTree(*datasetInfo, numClasses,
          successProbability, maxSamples, checkInterval, minSamples,
          CategoricalSplitType<FitnessFunction>(0, numClasses),
          numericSplits[0], dimensionMappings, false));
    }
    else if (numericSplits.size() == 0)
    {
      // Pass a default numeric split.
      children.push_back(new HoeffdingTree(*datasetInfo, numClasses,
          successProbability, maxSamples, checkInterval, minSamples,
          categoricalSplits[0], NumericSplitType<FitnessFunction>(numClasses),
          dimensionMappings, false));
    }
    else
    {
      // Pass both splits that we already have.
      children.push_back(new HoeffdingTree(*datasetInfo, numClasses,
          successProbability, maxSamples, checkInterval, minSamples,
          categoricalSplits[0], numericSplits[0], dimensionMappings, false));
    }

    children[i]->MajorityClass() = childMajorities[i];
  }

  // Eliminate now-unnecessary split information.
  numericSplits.clear();
  categoricalSplits.clear();
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Reset()
{
  ResetTree();
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Reset(const size_t dimensionality, const size_t numClasses)
{
  if (ownsInfo)
    delete datasetInfo;
  datasetInfo = new data::DatasetInfo(dimensionality); // All features numeric.
  ownsInfo = true;

  this->numClasses = numClasses;

  ResetTree();
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Reset(const data::DatasetInfo& info, const size_t numClasses)
{
  if (ownsInfo)
    delete datasetInfo;
  datasetInfo = &info;
  ownsInfo = false;

  this->numClasses = numClasses;

  ResetTree();
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename Archive>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(splitDimension));

  // Clear memory for the mappings if necessary.
  if (cereal::is_loading<Archive>() && ownsMappings && dimensionMappings)
    delete dimensionMappings;

  ar(CEREAL_POINTER(dimensionMappings));

  // Special handling for const object.
  data::DatasetInfo* d = NULL;
  if (cereal::is_saving<Archive>())
    d = const_cast<data::DatasetInfo*>(datasetInfo);
  ar(CEREAL_POINTER(d));

  if (cereal::is_loading<Archive>())
  {
    if (datasetInfo && ownsInfo)
      delete datasetInfo;

    datasetInfo = d;
    ownsInfo = true;
    ownsMappings = true; // We also own the mappings we loaded.

    // Clear the children.
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];
    children.clear();
  }

  ar(CEREAL_NVP(majorityClass));
  ar(CEREAL_NVP(majorityProbability));

  // Depending on whether or not we have split yet, we may need to save
  // different things.
  if (splitDimension == size_t(-1))
  {
    // We have not yet split.  So we have to serialize the splits.
    ar(CEREAL_NVP(numSamples));
    ar(CEREAL_NVP(numClasses));
    ar(CEREAL_NVP(maxSamples));
    ar(CEREAL_NVP(successProbability));

    // Serialize the splits, but not if we haven't seen any samples yet (in
    // which case we can just reinitialize).
    if (cereal::is_loading<Archive>())
    {
      // Re-initialize all of the splits.
      numericSplits.clear();
      categoricalSplits.clear();
      for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
      {
        if (datasetInfo->Type(i) == data::Datatype::categorical)
          categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
              datasetInfo->NumMappings(i), numClasses));
        else
          numericSplits.push_back(
              NumericSplitType<FitnessFunction>(numClasses));
      }

      // Clear things we don't need.
      categoricalSplit = typename CategoricalSplitType<FitnessFunction>::
          SplitInfo(numClasses);
      numericSplit = typename NumericSplitType<FitnessFunction>::SplitInfo();
    }

    // There's no need to serialize if there's no information contained in the
    // splits.
    if (numSamples == 0)
      return;

    // Serialize numeric splits.
    ar(CEREAL_NVP(numericSplits));

    // Serialize categorical splits.
    ar(CEREAL_NVP(categoricalSplits));
  }
  else
  {
    // We have split, so we only need to save the split and the children.
    if (datasetInfo->Type(splitDimension) == data::Datatype::categorical)
      ar(CEREAL_NVP(categoricalSplit));
    else
      ar(CEREAL_NVP(numericSplit));

    // Serialize the children, because we have split.
      ar(CEREAL_VECTOR_POINTER(children));

    if (cereal::is_loading<Archive>())
    {
      for (size_t i = 0; i < children.size(); ++i)
      {
        // The child doesn't actually own its own DatasetInfo.  We do.  The same
        // applies for the dimension mappings.
        if (children[i]->datasetInfo == datasetInfo)
          children[i]->ownsInfo = false;
        children[i]->ownsMappings = false;
      }

      numericSplits.clear();
      categoricalSplits.clear();

      numSamples = 0;
      numClasses = 0;
      maxSamples = 0;
      successProbability = 0.0;
    }
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::TrainInternal(const MatType& data,
                 const arma::Row<size_t>& labels,
                 const bool batchTraining)
{
  if (batchTraining)
  {
    // Pass all the points through the nodes, and then split only after that.
    checkInterval = data.n_cols; // Only split on the last sample.
    // Don't split if there are fewer than five points.
    size_t oldMaxSamples = maxSamples;
    maxSamples = std::max(size_t(data.n_cols - 1), size_t(5));
    for (size_t i = 0; i < data.n_cols; ++i)
      Train(data.col(i), labels[i]);
    maxSamples = oldMaxSamples;

    // Now, if we did split, find out which points go to which child, and
    // perform the same batch training.
    if (children.size() > 0)
    {
      // We need to create a vector of indices that represent the points that
      // must go to each child, so we need children.size() vectors, but we don't
      // know how long they will be.  Therefore, we will create vectors each of
      // size data.n_cols, but will probably not use all the memory we
      // allocated, and then pass subvectors to the submat() function.
      std::vector<arma::uvec> indices(children.size(), arma::uvec(data.n_cols));
      arma::Col<size_t> counts =
          zeros<arma::Col<size_t>>(children.size());

      for (size_t i = 0; i < data.n_cols; ++i)
      {
        size_t direction = CalculateDirection(data.col(i));
        size_t currentIndex = counts[direction];
        indices[direction][currentIndex] = i;
        counts[direction]++;
      }

      // Now pass each of these submatrices to the children to perform
      // batch-mode training.
      for (size_t i = 0; i < children.size(); ++i)
      {
        // If we don't have any points that go to the child in question, don't
        // train that child.
        if (counts[i] == 0)
          continue;

        // The submatrix here is non-contiguous, but I think this will be faster
        // than copying the points to an ordered state.  We still have to
        // assemble the labels vector, though.
        arma::Row<size_t> childLabels = labels.cols(
            indices[i].subvec(0, counts[i] - 1));

        // Unfortunately, limitations of Armadillo's non-contiguous subviews
        // prohibits us from successfully passing the non-contiguous subview to
        // Train(), since the col() function is not provided.  So,
        // unfortunately, instead, we'll just extract the non-contiguous
        // submatrix.
        //
        // I'd rather be able to use:
        //
        // arma::Mat<typename MatType::elem_type> childData =
        //     data.cols(indices[i].subvec(0, counts[i] - 1));
        //
        // but this isn't currently supported by Armadillo.
        arma::Mat<typename MatType::elem_type> childData(data.n_rows,
            counts[i]);
        for (size_t j = 0; j < counts[i]; ++j)
          childData.col(j) = data.col(indices[i][j]);

        children[i]->Train(childData, childLabels, numClasses, true);
      }
    }
  }
  else
  {
    // We aren't training in batch mode; loop through the points.
    for (size_t i = 0; i < data.n_cols; ++i)
      Train(data.col(i), labels[i]);
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::ResetTree(const CategoricalSplitType<FitnessFunction>& categoricalSplitIn,
             const NumericSplitType<FitnessFunction>& numericSplitIn)
{
  // Generate mappings.
  if (ownsMappings)
    delete dimensionMappings;

  categoricalSplits.clear();
  numericSplits.clear();

  dimensionMappings =
      new std::unordered_map<size_t, std::pair<size_t, size_t>>();
  ownsMappings = true;
  for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
  {
    if (datasetInfo->Type(i) == data::Datatype::categorical)
    {
      categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
          datasetInfo->NumMappings(i), numClasses, categoricalSplitIn));
      (*dimensionMappings)[i] = std::make_pair(data::Datatype::categorical,
          categoricalSplits.size() - 1);
    }
    else
    {
      numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses,
          numericSplitIn));
      (*dimensionMappings)[i] = std::make_pair(data::Datatype::numeric,
          numericSplits.size() - 1);
    }
  }

  // Clear children.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Reset statistics.
  numSamples = 0;
  splitDimension = size_t(-1);
  majorityClass = 0;
  majorityProbability = 0.0;
  categoricalSplit =
      typename CategoricalSplitType<FitnessFunction>::SplitInfo(0);
  numericSplit = typename NumericSplitType<FitnessFunction>::SplitInfo();
}

} // namespace mlpack

#endif
