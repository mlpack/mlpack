/**
 * @file methods/decision_tree/decision_tree_regressor_impl.hpp
 * @author Rishabh Garg
 *
 * Implementation of decision tree regressor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_DECISION_TREE_REGRESSOR_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_DECISION_TREE_REGRESSOR_IMPL_HPP

#include "decision_tree_regressor.hpp"
#include "utils.hpp"

namespace mlpack {
namespace tree {

//! Construct, don't train.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor() :
    splitDimension(0),
    dimensionType(0),
    splitPointOrPrediction(0.0)
{
  // Nothing to do here.
}

//! Construct and train without weight.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector)
{
  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels, 0,
      weights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Construct and train without weight on numeric data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    LabelsType labels,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector)
{
  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false>(tmpData, 0, tmpData.n_cols, tmpLabels, 0, weights,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

//! Construct and train with weights.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<arma::is_arma_type<
        typename std::remove_reference<WeightsType>::type>::value>*)
{
  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;
  using TrueWeightsType = typename std::decay<WeightsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels, 0,
      tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Construct and train on numeric data with weights.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    LabelsType labels,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<
        arma::is_arma_type<
        typename std::remove_reference<
        WeightsType>::type>::value>*)
{
  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;
  using TrueWeightsType = typename std::decay<WeightsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, tmpLabels, 0, tmpWeights,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

//! Take ownership of another tree and train with weights.
template<typename FitnessFunction,
        template<typename> class NumericSplitType,
        template<typename> class CategoricalSplitType,
        typename DimensionSelectionType,
        bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    const DecisionTreeRegressor& other,
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const std::enable_if_t<arma::is_arma_type<
        typename std::remove_reference<WeightsType>::type>::value>*):
        NumericAuxiliarySplitInfo(other),
        CategoricalAuxiliarySplitInfo(other)
{
  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;
  using TrueWeightsType = typename std::decay<WeightsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));
  TrueWeightsType tmpWeights(std::move(weights));

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels, 0,
              tmpWeights, minimumLeafSize, minimumGainSplit);
}

//! Take ownership of another tree and train with weights.
template<typename FitnessFunction,
        template<typename> class NumericSplitType,
        template<typename> class CategoricalSplitType,
        typename DimensionSelectionType,
        bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    const DecisionTreeRegressor& other,
    MatType data,
    LabelsType labels,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<arma::is_arma_type<
        typename std::remove_reference<
        WeightsType>::type>::value>*):
        NumericAuxiliarySplitInfo(other),
        CategoricalAuxiliarySplitInfo(other)  // other info does need to copy
{
  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;
  using TrueWeightsType = typename std::decay<WeightsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, tmpLabels, 0, tmpWeights,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

//! Copy another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
DecisionTreeRegressor<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             NoRecursion
>::DecisionTreeRegressor(
    const DecisionTreeRegressor& other) :
    NumericAuxiliarySplitInfo(other),
    CategoricalAuxiliarySplitInfo(other),
    splitDimension(other.splitDimension),
    dimensionType(other.dimensionType),
    splitPointOrPrediction(other.splitPointOrPrediction)
{
  // Copy each child.
  for (size_t i = 0; i < other.children.size(); ++i)
    children.push_back(new DecisionTreeRegressor(*other.children[i]));
}

//! Take ownership of another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
DecisionTreeRegressor<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             NoRecursion
>::DecisionTreeRegressor(
    DecisionTreeRegressor&& other) :
    NumericAuxiliarySplitInfo(std::move(other)),
    CategoricalAuxiliarySplitInfo(std::move(other)),
    children(std::move(other.children)),
    splitDimension(other.splitDimension),
    dimensionType(other.dimensionType),
    splitPointOrPrediction(other.splitPointOrPrediction)
{
  // Nothing to do here.
}

//! Copy another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>&
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion
>::operator=(const DecisionTreeRegressor& other)
{
  if (this == &other)
    return *this; // Nothing to copy.

  // Clean memory if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Copy everything from the other tree.
  splitDimension = other.splitDimension;
  dimensionType = other.dimensionType;
  splitPointOrPrediction = other.splitPointOrPrediction;

  // Copy the children.
  for (size_t i = 0; i < other.children.size(); ++i)
    children.push_back(new DecisionTreeRegressor(*other.children[i]));

  // Copy the auxiliary info.
  NumericAuxiliarySplitInfo::operator=(other);
  CategoricalAuxiliarySplitInfo::operator=(other);

  return *this;
}

//! Take ownership of another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>&
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion
>::operator=(DecisionTreeRegressor&& other)
{
  if (this == &other)
    return *this; // Nothing to move.

  // Clean memory if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Take ownership of the other tree's components.
  children = std::move(other.children);
  splitDimension = other.splitDimension;
  dimensionType = other.dimensionType;
  splitPointOrPrediction = other.splitPointOrPrediction;

  // Take ownership of the auxiliary info.
  NumericAuxiliarySplitInfo::operator=(std::move(other));
  CategoricalAuxiliarySplitInfo::operator=(std::move(other));

  return *this;
}

//! Clean up memory.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
DecisionTreeRegressor<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             NoRecursion>::~DecisionTreeRegressor()
{
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
}

//! Train on the given data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector)
{
  // Sanity check on data.
  util::CheckSameSizes(data, labels, "DecisionTreeRegressor::Train()");

  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  return Train<false>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels,
      0, weights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Train on the given data, assuming all dimensions are numeric.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    LabelsType labels,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector)
{
  // Sanity check on data.
  util::CheckSameSizes(data, labels, "DecisionTreeRegressor::Train()");

  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  return Train<false>(tmpData, 0, tmpData.n_cols, tmpLabels, 0,
      weights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Train on the given weighted data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<
        arma::is_arma_type<
        typename std::remove_reference<
        WeightsType>::type>::value>*)
{
  // Sanity check on data.
  util::CheckSameSizes(data, labels, "DecisionTreeRegressor::Train()");

  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;
  using TrueWeightsType = typename std::decay<WeightsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  return Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels,
      0, tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Train on the given weighted all numeric data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    LabelsType labels,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<
        arma::is_arma_type<
        typename std::remove_reference<
        WeightsType>::type>::value>*)
{
  // Sanity check on data.
  util::CheckSameSizes(data, labels, "DecisionTreeRegressor::Train()");

  using TrueMatType = typename std::decay<MatType>::type;
  using TrueLabelsType = typename std::decay<LabelsType>::type;
  using TrueWeightsType = typename std::decay<WeightsType>::type;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueLabelsType tmpLabels(std::move(labels));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  return Train<true>(tmpData, 0, tmpData.n_cols, tmpLabels, 0,
      tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Train on the given data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<bool UseWeights, typename MatType, typename LabelsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType& data,
    const size_t begin,
    const size_t count,
    const data::DatasetInfo& datasetInfo,
    LabelsType& labels,
    const size_t numClasses,
    arma::rowvec& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType& dimensionSelector)
{
  // Clear children if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Look through the list of dimensions and obtain the gain of the best split.
  // We'll cache the best numeric and categorical split auxiliary information in
  // numericAux and categoricalAux (and clear them later if we make no split),
  double bestGain = FitnessFunction::template Evaluate<UseWeights>(
      labels.subvec(begin, begin + count - 1),
      numClasses,
      UseWeights ? weights.subvec(begin, begin + count - 1) : weights);
  size_t bestDim = datasetInfo.Dimensionality(); // This means "no split".
  const size_t end = dimensionSelector.End();

  if (maximumDepth != 1)
  {
    for (size_t i = dimensionSelector.Begin(); i != end;
         i = dimensionSelector.Next())
    {
      double dimGain = -DBL_MAX;
      if (datasetInfo.Type(i) == data::Datatype::categorical)
      {
        dimGain = CategoricalSplit::template SplitIfBetter<UseWeights>(bestGain,
            data.cols(begin, begin + count - 1).row(i),
            datasetInfo.NumMappings(i),
            labels.subvec(begin, begin + count - 1),
            numClasses,
            UseWeights ? weights.subvec(begin, begin + count - 1) : weights,
            minimumLeafSize,
            minimumGainSplit,
            splitPointOrPrediction,
            *this);
      }
      else if (datasetInfo.Type(i) == data::Datatype::numeric)
      {
        dimGain = NumericSplit::template SplitIfBetter<UseWeights>(bestGain,
            data.cols(begin, begin + count - 1).row(i),
            labels.subvec(begin, begin + count - 1),
            UseWeights ? weights.subvec(begin, begin + count - 1) : weights,
            minimumLeafSize,
            minimumGainSplit,
            splitPointOrPrediction,
            *this);
      }

      // If the splitter reported that it did not split, move to the next
      // dimension.
      if (dimGain == DBL_MAX)
        continue;

      // Was there an improvement?  If so mark that it's the new best dimension.
      bestDim = i;
      bestGain = dimGain;

      // If the gain is the best possible, no need to keep looking.
      if (bestGain >= 0.0)
        break;
    }
  }

  // Did we split or not?  If so, then split the data and create the children.
  if (bestDim != datasetInfo.Dimensionality())
  {
    dimensionType = (size_t) datasetInfo.Type(bestDim);
    splitDimension = bestDim;

    // Get the number of children we will have.
    size_t numChildren = 0;
    if (datasetInfo.Type(bestDim) == data::Datatype::categorical)
      numChildren = CategoricalSplit::NumChildren(splitPointOrPrediction, *this);
    else
      numChildren = NumericSplit::NumChildren(splitPointOrPrediction, *this);

    // Calculate all child assignments.
    arma::Row<size_t> childAssignments(count);
    if (datasetInfo.Type(bestDim) == data::Datatype::categorical)
    {
      for (size_t j = begin; j < begin + count; ++j)
        childAssignments[j - begin] = CategoricalSplit::CalculateDirection(
            data(bestDim, j), splitPointOrPrediction, *this);
    }
    else
    {
      for (size_t j = begin; j < begin + count; ++j)
      {
        childAssignments[j - begin] = NumericSplit::CalculateDirection(
            data(bestDim, j), splitPointOrPrediction, *this);
      }
    }

    // Figure out counts of children.
    arma::Row<size_t> childCounts(numChildren, arma::fill::zeros);
    for (size_t i = begin; i < begin + count; ++i)
      childCounts[childAssignments[i - begin]]++;

    // Initialize bestGain if recursive split is allowed.
    if (!NoRecursion)
    {
      bestGain = 0.0;
    }

    // Split into children.
    size_t currentCol = begin;
    for (size_t i = 0; i < numChildren; ++i)
    {
      size_t currentChildBegin = currentCol;
      for (size_t j = currentChildBegin; j < begin + count; ++j)
      {
        if (childAssignments[j - begin] == i)
        {
          childAssignments.swap_cols(currentCol - begin, j - begin);
          data.swap_cols(currentCol, j);
          labels.swap_cols(currentCol, j);
          if (UseWeights)
            weights.swap_cols(currentCol, j);
          ++currentCol;
        }
      }

      // Now build the child recursively.
      DecisionTreeRegressor* child = new DecisionTreeRegressor();
      if (NoRecursion)
      {
        child->Train<UseWeights>(data, currentChildBegin,
            currentCol - currentChildBegin, datasetInfo, labels, numClasses,
            weights, currentCol - currentChildBegin, minimumGainSplit,
            maximumDepth - 1, dimensionSelector);
      }
      else
      {
        // During recursion entropy of child node may change.
        double childGain = child->Train<UseWeights>(data, currentChildBegin,
            currentCol - currentChildBegin, datasetInfo, labels, numClasses,
            weights, minimumLeafSize, minimumGainSplit, maximumDepth - 1,
            dimensionSelector);
        bestGain += double(childCounts[i]) / double(count) * (-childGain);
      }
      children.push_back(child);
    }
  }
  else
  {
    // Clear auxiliary info objects.
    NumericAuxiliarySplitInfo::operator=(NumericAuxiliarySplitInfo());
    CategoricalAuxiliarySplitInfo::operator=(CategoricalAuxiliarySplitInfo());

    // Calculate prediction label because we are a leaf.
    CalculatePrediction<UseWeights>(
        labels.subvec(begin, begin + count - 1),
        UseWeights ? weights.subvec(begin, begin + count - 1) : weights);   
    std::cout << "Number of points in leaf: " << count << 
        "  Prediction: " << splitPointOrPrediction << std::endl; 
  }

  return -bestGain;
}

//! Train on the given data, assuming all dimensions are numeric.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<bool UseWeights, typename MatType, typename LabelsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType& data,
    const size_t begin,
    const size_t count,
    LabelsType& labels,
    const size_t numClasses,
    arma::rowvec& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType& dimensionSelector)
{
  // Clear children if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // We won't be using these members, so reset them.
  CategoricalAuxiliarySplitInfo::operator=(CategoricalAuxiliarySplitInfo());

  // Look through the list of dimensions and obtain the best split.  We'll cache
  // the best numeric split auxiliary information in numericAux (and clear it
  // later if we don't make a split), and use classProbabilities as auxiliary
  // information.  Later we'll overwrite classProbabilities to the empirical
  // class probabilities if we do not split.
  double bestGain = FitnessFunction::template Evaluate<UseWeights>(
      labels.subvec(begin, begin + count - 1),
      numClasses,
      UseWeights ? weights.subvec(begin, begin + count - 1) : weights);
  size_t bestDim = data.n_rows; // This means "no split".

  if (maximumDepth != 1)
  {
    for (size_t i = dimensionSelector.Begin(); i != dimensionSelector.End();
         i = dimensionSelector.Next())
    {
      const double dimGain = NumericSplitType<FitnessFunction>::template
          SplitIfBetter<UseWeights>(bestGain,
                                    data.cols(begin, begin + count - 1).row(i),
                                    labels.cols(begin, begin + count - 1),
                                    UseWeights ?
                                        weights.cols(begin, begin + count - 1) :
                                        weights,
                                    minimumLeafSize,
                                    minimumGainSplit,
                                    splitPointOrPrediction,
                                    *this);

      // If the splitter did not report that it improved, then move to the next
      // dimension.
      // std::cout << "dim: " << i << "   gain: " << dimGain << std::endl;
      if (dimGain == DBL_MAX)
        continue;

      bestDim = i;
      bestGain = dimGain;

      // If the gain is the best possible, no need to keep looking.
      if (bestGain >= 0.0)
        break;
    }
  }

  // Did we split or not?  If so, then split the data and create the children.
  if (bestDim != data.n_rows)
  {
    // std::cout << "splitpoint: " << splitPointOrPrediction << std::endl;
    // std::cout << "bestDim: " << bestDim << std::endl;
    // We know that the split is numeric.
    size_t numChildren = NumericSplit::NumChildren(splitPointOrPrediction, *this);
    splitDimension = bestDim;
    dimensionType = (size_t) data::Datatype::numeric;

    // Calculate all child assignments.
    arma::Row<size_t> childAssignments(count);

    for (size_t j = begin; j < begin + count; ++j)
    {
      childAssignments[j - begin] = NumericSplit::CalculateDirection(
          data(bestDim, j), splitPointOrPrediction, *this);
    }

    // Calculate counts of children in each node.
    arma::Row<size_t> childCounts(numChildren);
    childCounts.zeros();
    for (size_t j = begin; j < begin + count; ++j)
      childCounts[childAssignments[j - begin]]++;

    // Initialize bestGain if recursive split is allowed.
    if (!NoRecursion)
    {
      bestGain = 0.0;
    }

    size_t currentCol = begin;
    for (size_t i = 0; i < numChildren; ++i)
    {
      size_t currentChildBegin = currentCol;
      for (size_t j = currentChildBegin; j < begin + count; ++j)
      {
        if (childAssignments[j - begin] == i)
        {
          childAssignments.swap_cols(currentCol - begin, j - begin);
          data.swap_cols(currentCol, j);
          labels.swap_cols(currentCol, j);
          if (UseWeights)
            weights.swap_cols(currentCol, j);
          ++currentCol;
        }
      }

      // Now build the child recursively.
      DecisionTreeRegressor* child = new DecisionTreeRegressor();
      if (NoRecursion)
      {
        child->Train<UseWeights>(data, currentChildBegin,
            currentCol - currentChildBegin, labels, numClasses, weights,
            currentCol - currentChildBegin, minimumGainSplit, maximumDepth - 1,
            dimensionSelector);
      }
      else
      {
        // During recursion entropy of child node may change.
        double childGain = child->Train<UseWeights>(data, currentChildBegin,
            currentCol - currentChildBegin, labels, numClasses, weights,
            minimumLeafSize, minimumGainSplit, maximumDepth - 1,
            dimensionSelector);
        bestGain += double(childCounts[i]) / double(count) * (-childGain);
      }
      children.push_back(child);
    }
  }
  else
  {
    // We won't be needing these members, so reset them.
    NumericAuxiliarySplitInfo::operator=(NumericAuxiliarySplitInfo());

    // Calculate prediction label because we are a leaf.
    CalculatePrediction<UseWeights>(
        labels.subvec(begin, begin + count - 1),
        UseWeights ? weights.subvec(begin, begin + count - 1) : weights);
    std::cout << "Number of points in leaf: " << count << 
        "  Prediction: " << splitPointOrPrediction << std::endl;
  }

  return -bestGain;
}

//! Return the prediction.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename VecType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Predict(const VecType& point) const
{
  if (children.size() == 0)
  {
    // Return cached prediction.
    return splitPointOrPrediction;
  }

  return children[CalculateDirection(point)]->Predict(point);
}

//! Return the predictions for a set of points.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType>
void DecisionTreeRegressor<FitnessFunction,
                           NumericSplitType,
                           CategoricalSplitType,
                           DimensionSelectionType,
                           NoRecursion
>::Predict(const MatType& data, arma::Row<double>& predictions) const
{
  predictions.set_size(data.n_cols);
  // If the tree's root is leaf.
  if (children.size() == 0)
  {
    predictions.fill(splitPointOrPrediction);
    return;
  }

  // Loop over each point.
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Predict(data.col(i));
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<bool UseWeights, typename LabelsType, typename WeightsType>
void DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion
>::CalculatePrediction(const LabelsType& labels, const WeightsType& weights)
{
  if (UseWeights)
  {
    double accWeights, weightedSum;
    WeightedSum(labels, weights, 0, labels.n_elem, accWeights, weightedSum);
    splitPointOrPrediction = weightedSum / accWeights;
  }
  else
  {
    double sum;
    Sum(labels, 0, labels.n_elem, sum);
    splitPointOrPrediction = sum / labels.n_elem;
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename VecType>
size_t DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion
>::CalculateDirection(const VecType& point) const
{
  if ((data::Datatype) dimensionType == data::Datatype::categorical)
    return CategoricalSplit::CalculateDirection(point[splitDimension],
        splitPointOrPrediction, *this);
  else
    return NumericSplit::CalculateDirection(point[splitDimension],
        splitPointOrPrediction, *this);
}

//! Serialize the tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename Archive>
void DecisionTreeRegressor<FitnessFunction,
                           NumericSplitType,
                           CategoricalSplitType,
                           DimensionSelectionType,
                           NoRecursion
>::serialize(Archive& ar, const uint32_t /* version */)
{
  // Clean memory if needed.
  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];
    children.clear();
  }
  // Serialize the children first.
  ar(CEREAL_VECTOR_POINTER(children));

  // Now serialize the rest of the object.
  ar(CEREAL_NVP(splitDimension));
  ar(CEREAL_NVP(dimensionType));
  ar(CEREAL_NVP(splitPointOrPrediction));
}

//! Return the number of leaves.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
size_t DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::NumLeaves() const
{
  if (this->NumChildren() == 0)
    return 1;

  size_t numLeaves = 0;
  for (size_t i = 0; i < this->NumChildren(); ++i)
    numLeaves += children[i]->NumLeaves();

  return numLeaves;
}

} // namespace tree
} // namespace mlpack

#endif
