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
    dimensionTypeOrMajorityClass(0),
    classProbabilities(numClasses)
{
  // Initialize utility vector.
  classProbabilities.fill(1.0 / (double) numClasses);
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
  Train<false>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels, numClasses,
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
  Train<false>(tmpData, 0, tmpData.n_cols, tmpLabels, numClasses, weights,
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
  Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels, numClasses,
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
  Train<true>(tmpData, 0, tmpData.n_cols, tmpLabels, numClasses, tmpWeights,
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
    const size_t numClasses,
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
  Train<true>(tmpData, 0, tmpData.n_cols, datasetInfo, tmpLabels, numClasses,
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
  Train<true>(tmpData, 0, tmpData.n_cols, tmpLabels, numClasses, tmpWeights,
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
    dimensionTypeOrMajorityClass(other.dimensionTypeOrMajorityClass),
    classProbabilities(other.classProbabilities)
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
    dimensionTypeOrMajorityClass(other.dimensionTypeOrMajorityClass),
    classProbabilities(std::move(other.classProbabilities))
{
  // Reset the other object.
  other.classProbabilities.ones(1); // One class, P(1) = 1.
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
  dimensionTypeOrMajorityClass = other.dimensionTypeOrMajorityClass;
  classProbabilities = other.classProbabilities;

  // Copy the children.
  for (size_t i = 0; i < other.children.size(); ++i)
    children.push_back(new DecisionTree(*other.children[i]));

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
  dimensionTypeOrMajorityClass = other.dimensionTypeOrMajorityClass;
  classProbabilities = std::move(other.classProbabilities);

  // Reset the class probabilities of the other object.
  other.classProbabilities.ones(1); // One class, P(1) = 1.

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
      numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth,
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
  return Train<false>(tmpData, 0, tmpData.n_cols, tmpLabels, numClasses,
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
      numClasses, tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
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
  return Train<true>(tmpData, 0, tmpData.n_cols, tmpLabels, numClasses,
      tmpWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}


} // namespace tree
} // namespace mlpack

#endif
