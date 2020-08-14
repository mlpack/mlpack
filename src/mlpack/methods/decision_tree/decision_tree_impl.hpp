/**
 * @file methods/decision_tree/decision_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of generic decision tree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_DECISION_TREE_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_DECISION_TREE_IMPL_HPP

#include "decision_tree.hpp"

namespace mlpack {
namespace tree {

//! Construct and train without weight.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::DecisionTree(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    const size_t numClasses,
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

//! Construct and train.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::DecisionTree(
    MatType data,
    LabelsType labels,
    const size_t numClasses,
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
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::DecisionTree(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    const size_t numClasses,
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

//! Construct and train with weights.
template<typename FitnessFunction,
        template<typename> class NumericSplitType,
        template<typename> class CategoricalSplitType,
        typename DimensionSelectionType,
        typename ElemType,
        bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTree<FitnessFunction,
        NumericSplitType,
        CategoricalSplitType,
        DimensionSelectionType,
        ElemType,
        NoRecursion>::DecisionTree(
    const DecisionTree& other,
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

//! Construct and train with weights.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::DecisionTree(
    MatType data,
    LabelsType labels,
    const size_t numClasses,
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

//! Construct and train with weights.
template<typename FitnessFunction,
        template<typename> class NumericSplitType,
        template<typename> class CategoricalSplitType,
        typename DimensionSelectionType,
        typename ElemType,
        bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTree<FitnessFunction,
        NumericSplitType,
        CategoricalSplitType,
        DimensionSelectionType,
        ElemType,
        NoRecursion>::DecisionTree(
    const DecisionTree& other,
    MatType data,
    LabelsType labels,
    const size_t numClasses,
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

//! Construct, don't train.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::DecisionTree(const size_t numClasses) :
    splitDimension(0),
    dimensionTypeOrMajorityClass(0),
    classProbabilities(numClasses)
{
  // Initialize utility vector.
  classProbabilities.fill(1.0 / (double) numClasses);
}

//! Copy another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::DecisionTree(const DecisionTree& other) :
    NumericAuxiliarySplitInfo(other),
    CategoricalAuxiliarySplitInfo(other),
    splitDimension(other.splitDimension),
    dimensionTypeOrMajorityClass(other.dimensionTypeOrMajorityClass),
    classProbabilities(other.classProbabilities)
{
  // Copy each child.
  for (size_t i = 0; i < other.children.size(); ++i)
    children.push_back(new DecisionTree(*other.children[i]));
}

//! Take ownership of another tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::DecisionTree(DecisionTree&& other) :
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
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>&
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::operator=(const DecisionTree& other)
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
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>&
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::operator=(DecisionTree&& other)
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
         typename ElemType,
         bool NoRecursion>
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             ElemType,
             NoRecursion>::~DecisionTree()
{
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
}

//! Train on the given data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
double DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::Train(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    const size_t numClasses,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector)
{
  // Sanity check on data.
  if (data.n_cols != labels.n_elem)
  {
    std::ostringstream oss;
    oss << "DecisionTree::Train(): number of points (" << data.n_cols << ") "
        << "does not match number of labels (" << labels.n_elem << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }

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
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType>
double DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::Train(
    MatType data,
    LabelsType labels,
    const size_t numClasses,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    const size_t maximumDepth,
    DimensionSelectionType dimensionSelector)
{
  // Sanity check on data.
  if (data.n_cols != labels.n_elem)
  {
    std::ostringstream oss;
    oss << "DecisionTree::Train(): number of points (" << data.n_cols << ") "
        << "does not match number of labels (" << labels.n_elem << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }

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
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
double DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::Train(
    MatType data,
    const data::DatasetInfo& datasetInfo,
    LabelsType labels,
    const size_t numClasses,
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
  if (data.n_cols != labels.n_elem)
  {
    std::ostringstream oss;
    oss << "DecisionTree::Train(): number of points (" << data.n_cols << ") "
        << "does not match number of labels (" << labels.n_elem << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }

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

//! Train on the given weighted data.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType, typename LabelsType, typename WeightsType>
double DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::Train(
    MatType data,
    LabelsType labels,
    const size_t numClasses,
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
  if (data.n_cols != labels.n_elem)
  {
    std::ostringstream oss;
    oss << "DecisionTree::Train(): number of points (" << data.n_cols << ") "
        << "does not match number of labels (" << labels.n_elem << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }

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

//! Train on the given data, assuming all dimensions are numeric.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<bool UseWeights, typename MatType>
double DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::Train(
    MatType& data,
    const size_t begin,
    const size_t count,
    const data::DatasetInfo& datasetInfo,
    arma::Row<size_t>& labels,
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
  // and use classProbabilities as auxiliary information.  Later we'll overwrite
  // classProbabilities to the empirical class probabilities if we do not split.
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
            classProbabilities,
            *this);
      }
      else if (datasetInfo.Type(i) == data::Datatype::numeric)
      {
        dimGain = NumericSplit::template SplitIfBetter<UseWeights>(bestGain,
            data.cols(begin, begin + count - 1).row(i),
            labels.subvec(begin, begin + count - 1),
            numClasses,
            UseWeights ? weights.subvec(begin, begin + count - 1) : weights,
            minimumLeafSize,
            minimumGainSplit,
            classProbabilities,
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
    dimensionTypeOrMajorityClass = (size_t) datasetInfo.Type(bestDim);
    splitDimension = bestDim;

    // Get the number of children we will have.
    size_t numChildren = 0;
    if (datasetInfo.Type(bestDim) == data::Datatype::categorical)
      numChildren = CategoricalSplit::NumChildren(classProbabilities, *this);
    else
      numChildren = NumericSplit::NumChildren(classProbabilities, *this);

    // Calculate all child assignments.
    arma::Row<size_t> childAssignments(count);
    if (datasetInfo.Type(bestDim) == data::Datatype::categorical)
    {
      for (size_t j = begin; j < begin + count; ++j)
        childAssignments[j - begin] = CategoricalSplit::CalculateDirection(
            data(bestDim, j), classProbabilities, *this);
    }
    else
    {
      for (size_t j = begin; j < begin + count; ++j)
      {
        childAssignments[j - begin] = NumericSplit::CalculateDirection(
            data(bestDim, j), classProbabilities, *this);
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
      DecisionTree* child = new DecisionTree();
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

    // Calculate class probabilities because we are a leaf.
    CalculateClassProbabilities<UseWeights>(
        labels.subvec(begin, begin + count - 1),
        numClasses,
        UseWeights ? weights.subvec(begin, begin + count - 1) : weights);
  }

  return -bestGain;
}

//! Train on the given data, assuming all dimensions are numeric.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<bool UseWeights, typename MatType>
double DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::Train(
    MatType& data,
    const size_t begin,
    const size_t count,
    arma::Row<size_t>& labels,
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
                                    numClasses,
                                    UseWeights ?
                                        weights.cols(begin, begin + count - 1) :
                                        weights,
                                    minimumLeafSize,
                                    minimumGainSplit,
                                    classProbabilities,
                                    *this);

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
    size_t numChildren = NumericSplit::NumChildren(classProbabilities, *this);
    splitDimension = bestDim;
    dimensionTypeOrMajorityClass = (size_t) data::Datatype::numeric;

    // Calculate all child assignments.
    arma::Row<size_t> childAssignments(count);

    for (size_t j = begin; j < begin + count; ++j)
    {
      childAssignments[j - begin] = NumericSplit::CalculateDirection(
          data(bestDim, j), classProbabilities, *this);
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
      DecisionTree* child = new DecisionTree();
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

    // Calculate class probabilities because we are a leaf.
    CalculateClassProbabilities<UseWeights>(
        labels.subvec(begin, begin + count - 1),
        numClasses,
        UseWeights ? weights.subvec(begin, begin + count - 1) : weights);
  }

  return -bestGain;
}

//! Return the class.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename VecType>
size_t DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::Classify(const VecType& point) const
{
  if (children.size() == 0)
  {
    // Return cached max of probabilities.
    return dimensionTypeOrMajorityClass;
  }

  return children[CalculateDirection(point)]->Classify(point);
}

//! Return class probabilities for a given point.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename VecType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  DimensionSelectionType,
                  ElemType,
                  NoRecursion>::Classify(const VecType& point,
                                         size_t& prediction,
                                         arma::vec& probabilities) const
{
  if (children.size() == 0)
  {
    prediction = dimensionTypeOrMajorityClass;
    probabilities = classProbabilities;
    return;
  }

  children[CalculateDirection(point)]->Classify(point, prediction,
      probabilities);
}

//! Return the class for a set of points.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  DimensionSelectionType,
                  ElemType,
                  NoRecursion>::Classify(const MatType& data,
                                         arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);
  if (children.size() == 0)
  {
    predictions.fill(dimensionTypeOrMajorityClass);
    return;
  }

  // Loop over each point.
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Classify(data.col(i));
}

//! Return the class probabilities for a set of points.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename MatType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  DimensionSelectionType,
                  ElemType,
                  NoRecursion>::Classify(const MatType& data,
                                         arma::Row<size_t>& predictions,
                                         arma::mat& probabilities) const
{
  predictions.set_size(data.n_cols);
  if (children.size() == 0)
  {
    predictions.fill(dimensionTypeOrMajorityClass);
    probabilities = arma::repmat(classProbabilities, 1, data.n_cols);
    return;
  }

  // Otherwise we have to find the right size to set the predictions matrix to
  // be.
  DecisionTree* node = children[0];
  while (node->NumChildren() != 0)
    node = &node->Child(0);
  probabilities.set_size(node->classProbabilities.n_elem, data.n_cols);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::vec v = probabilities.unsafe_col(i); // Alias of column.
    Classify(data.col(i), predictions[i], v);
  }
}

//! Serialize the tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename Archive>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  DimensionSelectionType,
                  ElemType,
                  NoRecursion>::serialize(Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  // Clean memory if needed.
  if (Archive::is_loading::value)
  {
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];
    children.clear();
  }
  // Serialize the children first.
  CEREAL_VECTOR_POINTER(children);

  // Now serialize the rest of the object.
  ar & CEREAL_NVP(splitDimension);
  ar & CEREAL_NVP(dimensionTypeOrMajorityClass);
  ar & CEREAL_NVP(classProbabilities);
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<typename VecType>
size_t DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::CalculateDirection(const VecType& point) const
{
  if ((data::Datatype) dimensionTypeOrMajorityClass ==
      data::Datatype::categorical)
    return CategoricalSplit::CalculateDirection(point[splitDimension],
        classProbabilities, *this);
  else
    return NumericSplit::CalculateDirection(point[splitDimension],
        classProbabilities, *this);
}

// Get the number of classes in the tree.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
size_t DecisionTree<FitnessFunction,
                    NumericSplitType,
                    CategoricalSplitType,
                    DimensionSelectionType,
                    ElemType,
                    NoRecursion>::NumClasses() const
{
  // Recurse to the nearest child and return the number of elements in the
  // probability vector.
  if (children.size() == 0)
    return classProbabilities.n_elem;
  else
    return children[0]->NumClasses();
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         typename ElemType,
         bool NoRecursion>
template<bool UseWeights, typename RowType, typename WeightsRowType>
void DecisionTree<FitnessFunction,
                  NumericSplitType,
                  CategoricalSplitType,
                  DimensionSelectionType,
                  ElemType,
                  NoRecursion>::CalculateClassProbabilities(
    const RowType& labels,
    const size_t numClasses,
    const WeightsRowType& weights)
{
  classProbabilities.zeros(numClasses);
  double sumWeights = 0.0;
  for (size_t i = 0; i < labels.n_elem; ++i)
  {
    if (UseWeights)
    {
      classProbabilities[labels[i]] += weights[i];
      sumWeights += weights[i];
    }
    else
    {
      classProbabilities[labels[i]]++;
    }
  }

  // Now normalize into probabilities.
  classProbabilities /= UseWeights ? sumWeights : labels.n_elem;
  arma::uword maxIndex = 0;
  classProbabilities.max(maxIndex);
  dimensionTypeOrMajorityClass = (size_t) maxIndex;
}

} // namespace tree
} // namespace mlpack

#endif
