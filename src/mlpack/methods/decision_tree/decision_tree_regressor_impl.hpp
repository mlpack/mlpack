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
                      NoRecursion>::DecisionTreeRegressor()
{
  // Nothing to do here.
}

//! Construct and train without weight.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    ResponsesType responses,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector) : splitInfo()
{
  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpResponses,
      weights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Construct and train without weight on numeric data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    ResponsesType responses,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector) : splitInfo()
{
  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false>(tmpData, 0, tmpData.n_cols, tmpResponses, weights,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

//! Construct and train with weights.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    ResponsesType responses,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<arma::is_arma_type<
        std::remove_reference_t<WeightsType>>::value>*)
    : splitInfo()
{
  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;
  using TrueWeightsType = std::decay_t<WeightsType>;

  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpResponses,
      tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

//! Construct and train on numeric data with weights.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    MatType data,
    ResponsesType responses,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<
        arma::is_arma_type<std::remove_reference_t<WeightsType>>::value>*) :
    splitInfo()
{
  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;
  using TrueWeightsType = std::decay_t<WeightsType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, tmpResponses, tmpWeights,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

//! Take ownership of another tree and train with weights.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    const DecisionTreeRegressor& other,
    MatType data,
    const data::DatasetInfo& datasetInfo,
    ResponsesType responses,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const std::enable_if_t<arma::is_arma_type<
        std::remove_reference_t<WeightsType>>::value>*) :
    splitInfo(std::move(other.splitInfo)),
    NumericAuxiliarySplitInfo(other),
    CategoricalAuxiliarySplitInfo(other)
{
  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;
  using TrueWeightsType = std::decay_t<WeightsType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));
  TrueWeightsType tmpWeights(std::move(weights));

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpResponses,
              tmpWeights, minimumLeafSize, minimumGainSplit);
}

//! Take ownership of another tree and train with weights.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType, typename WeightsType>
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::DecisionTreeRegressor(
    const DecisionTreeRegressor& other,
    MatType data,
    ResponsesType responses,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    const std::enable_if_t<arma::is_arma_type<
        std::remove_reference_t<WeightsType>>::value>*) :
    splitInfo(std::move(other.splitInfo)),
    NumericAuxiliarySplitInfo(other),
    CategoricalAuxiliarySplitInfo(other)   // other info does need to copy
{
  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;
  using TrueWeightsType = std::decay_t<WeightsType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the weighted Train() method.
  Train<true>(tmpData, 0, tmpData.n_cols, tmpResponses, tmpWeights,
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
    prediction(other.prediction),
    dimensionType(other.dimensionType)
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
    prediction(other.prediction),
    dimensionType(other.dimensionType),
    splitInfo(other.splitInfo),
    NumericAuxiliarySplitInfo(std::move(other)),
    CategoricalAuxiliarySplitInfo(std::move(other)),
    children(std::move(other.children))
{
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
  dimensionType = other.dimensionType;
  splitInfo = other.splitInfo;
  if (other.children.size() != 0)
    splitDimension = other.splitDimension;
  else
    prediction = other.prediction;

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
  dimensionType = other.dimensionType;
  splitInfo = other.splitInfo;

  if (children.size() != 0)
    splitDimension = other.splitDimension;
  else
    prediction = other.prediction;

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
template<typename MatType, typename ResponsesType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    ResponsesType responses,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    FitnessFunction fitnessFunction)
{
  // Sanity check on data.
  util::CheckSameSizes(data, responses, "DecisionTreeRegressor::Train()");

  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  return Train<false>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpResponses,
      weights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, fitnessFunction);
}

//! Train on the given data, assuming all dimensions are numeric.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    ResponsesType responses,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    FitnessFunction fitnessFunction)
{
  // Sanity check on data.
  util::CheckSameSizes(data, responses, "DecisionTreeRegressor::Train()");

  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  return Train<false>(tmpData, 0, tmpData.n_cols, tmpResponses,
      weights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, fitnessFunction);
}

//! Train on the given weighted data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType, typename WeightsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    ResponsesType responses,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    FitnessFunction fitnessFunction,
    const std::enable_if_t<
        arma::is_arma_type<std::remove_reference_t<WeightsType>>::value>*)
{
  // Sanity check on data.
  util::CheckSameSizes(data, responses, "DecisionTreeRegressor::Train()");

  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;
  using TrueWeightsType = std::decay_t<WeightsType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  return Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpResponses,
      tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, fitnessFunction);
}

//! Train on the given weighted all numeric data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename ResponsesType, typename WeightsType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType data,
    ResponsesType responses,
    WeightsType weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector,
    FitnessFunction fitnessFunction,
    const std::enable_if_t<
        arma::is_arma_type<std::remove_reference_t<WeightsType>>::value>*)
{
  // Sanity check on data.
  util::CheckSameSizes(data, responses, "DecisionTreeRegressor::Train()");

  using TrueMatType = std::decay_t<MatType>;
  using TrueResponsesType = std::decay_t<ResponsesType>;
  using TrueWeightsType = std::decay_t<WeightsType>;

  // Copy or move data.
  TrueMatType tmpData(std::move(data));
  TrueResponsesType tmpResponses(std::move(responses));
  TrueWeightsType tmpWeights(std::move(weights));

  // Set the correct dimensionality for the dimension selector.
  dimensionSelector.Dimensions() = tmpData.n_rows;

  // Pass off work to the Train() method.
  return Train<true>(tmpData, 0, tmpData.n_cols, tmpResponses,
      tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, fitnessFunction);
}

//! Train on the given data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<bool UseWeights, typename MatType, typename ResponsesType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType& data,
    const size_t begin,
    const size_t count,
    const data::DatasetInfo& datasetInfo,
    ResponsesType& responses,
    arma::rowvec& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType& dimensionSelector,
    FitnessFunction fitnessFunction)
{
  // Clear children if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // Look through the list of dimensions and obtain the gain of the best split.
  // We'll cache the best numeric and categorical split auxiliary information
  // in splitInfo, which is non-empty only for internal nodes of the tree.
  double bestGain = fitnessFunction.template Evaluate<UseWeights>(
      responses.cols(begin, begin + count - 1),
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
            responses.cols(begin, begin + count - 1),
            UseWeights ? weights.subvec(begin, begin + count - 1) : weights,
            minimumLeafSize,
            minimumGainSplit,
            splitInfo,
            *this,
            fitnessFunction);
      }
      else if (datasetInfo.Type(i) == data::Datatype::numeric)
      {
        dimGain = NumericSplit::template SplitIfBetter<UseWeights>(bestGain,
            data.cols(begin, begin + count - 1).row(i),
            responses.cols(begin, begin + count - 1),
            UseWeights ? weights.subvec(begin, begin + count - 1) : weights,
            minimumLeafSize,
            minimumGainSplit,
            splitInfo,
            *this,
            fitnessFunction);
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
      numChildren = CategoricalSplit::NumChildren(splitInfo, *this);
    else
      numChildren = NumericSplit::NumChildren(splitInfo, *this);

    // Calculate all child assignments.
    arma::Row<size_t> childAssignments(count);
    if (datasetInfo.Type(bestDim) == data::Datatype::categorical)
    {
      for (size_t j = begin; j < begin + count; ++j)
        childAssignments[j - begin] = CategoricalSplit::CalculateDirection(
            data(bestDim, j), splitInfo, *this);
    }
    else
    {
      for (size_t j = begin; j < begin + count; ++j)
      {
        childAssignments[j - begin] = NumericSplit::CalculateDirection(
            data(bestDim, j), splitInfo, *this);
      }
    }

    // Figure out counts of children.
    arma::Row<size_t> childCounts(numChildren);
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
          responses.swap_cols(currentCol, j);
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
            currentCol - currentChildBegin, datasetInfo, responses,
            weights, currentCol - currentChildBegin, minimumGainSplit,
            maximumDepth - 1, dimensionSelector);
      }
      else
      {
        // During recursion entropy of child node may change.
        double childGain = child->Train<UseWeights>(data, currentChildBegin,
            currentCol - currentChildBegin, datasetInfo, responses,
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

    // Calculate prediction value because we are a leaf.
    prediction = fitnessFunction.template OutputLeafValue<UseWeights>(
        responses.cols(begin, begin + count - 1),
        UseWeights ? weights.subvec(begin, begin + count - 1) : weights);
  }

  return -bestGain;
}

//! Train on the given data, assuming all dimensions are numeric.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<bool UseWeights, typename MatType, typename ResponsesType>
double DecisionTreeRegressor<FitnessFunction,
                             NumericSplitType,
                             CategoricalSplitType,
                             DimensionSelectionType,
                             NoRecursion>::Train(
    MatType& data,
    const size_t begin,
    const size_t count,
    ResponsesType& responses,
    arma::rowvec& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType& dimensionSelector,
    FitnessFunction fitnessFunction)
{
  // Clear children if needed.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  // We won't be using these members, so reset them.
  CategoricalAuxiliarySplitInfo::operator=(CategoricalAuxiliarySplitInfo());

  // Look through the list of dimensions and obtain the best split. We'll
  // cache the best numeric and categorical split auxiliary information
  // in splitInfo, which is non-empty only for internal nodes of the tree.
  // splitPointOrPrediction for all internal nodes of the tree.
  double bestGain = fitnessFunction.template Evaluate<UseWeights>(
      responses.cols(begin, begin + count - 1),
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
                                    responses.cols(begin, begin + count - 1),
                                    UseWeights ?
                                        weights.cols(begin, begin + count - 1) :
                                        weights,
                                    minimumLeafSize,
                                    minimumGainSplit,
                                    splitInfo,
                                    *this,
                                    fitnessFunction);

      // If the splitter did not report that it improved, then move to the next
      // dimension.
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
    // We know that the split is numeric.
    size_t numChildren = NumericSplit::NumChildren(splitInfo, *this);
    splitDimension = bestDim;
    dimensionType = (size_t) data::Datatype::numeric;

    // Calculate all child assignments.
    arma::Row<size_t> childAssignments(count);

    for (size_t j = begin; j < begin + count; ++j)
    {
      childAssignments[j - begin] = NumericSplit::CalculateDirection(
          data(bestDim, j), splitInfo, *this);
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
          responses.swap_cols(currentCol, j);
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
            currentCol - currentChildBegin, responses, weights,
            currentCol - currentChildBegin, minimumGainSplit, maximumDepth - 1,
            dimensionSelector);
      }
      else
      {
        // During recursion entropy of child node may change.
        double childGain = child->Train<UseWeights>(data, currentChildBegin,
            currentCol - currentChildBegin, responses, weights,
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

    // Calculate prediction value because we are a leaf.
    prediction = fitnessFunction.template OutputLeafValue<UseWeights>(
        responses.cols(begin, begin + count - 1),
        UseWeights ? weights.subvec(begin, begin + count - 1) : weights);
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
typename VecType::elem_type
DecisionTreeRegressor<FitnessFunction,
                      NumericSplitType,
                      CategoricalSplitType,
                      DimensionSelectionType,
                      NoRecursion>::Predict(const VecType& point) const
{
  if (children.size() == 0)
  {
    // Return cached prediction.
    return prediction;
  }

  using ElemType = typename VecType::elem_type;
  return (ElemType) children[CalculateDirection(point)]->Predict(point);
}

//! Return the predictions for a set of points.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
template<typename MatType, typename PredVecType>
void DecisionTreeRegressor<FitnessFunction,
                           NumericSplitType,
                           CategoricalSplitType,
                           DimensionSelectionType,
                           NoRecursion
>::Predict(const MatType& data,
           PredVecType& predictions) const
{
  using ElemType = typename PredVecType::elem_type;

  predictions.set_size(data.n_cols);
  // If the tree's root is leaf.
  if (children.size() == 0)
  {
    predictions.fill((ElemType) prediction);
    return;
  }

  // Loop over each point.
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = (ElemType) Predict(data.col(i));
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
        splitInfo, *this);
  else
    return NumericSplit::CalculateDirection(point[splitDimension],
        splitInfo, *this);
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

  // Now serialize the rest of the object. Since splitDimension and
  // prediction are a union, we only need to serialize one of them.
  ar(CEREAL_NVP(prediction));
  ar(CEREAL_NVP(dimensionType));
  ar(CEREAL_NVP(splitInfo));
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

} // namespace mlpack

#endif
