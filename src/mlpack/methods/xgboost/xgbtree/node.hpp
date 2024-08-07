/**
 * @file methods/xgboost/xgbtree/node.hpp
 * @author Abhimanyu Dayal
 *
 * XGB tree node.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBTREE_NODE_HPP
#define MLPACK_METHODS_XGBTREE_NODE_HPP

#include "dt_prereq.hpp"

// Defined within the mlpack namespace.
namespace mlpack {

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
class Node
{
  Node() { /*Nothing to do*/ }

  Node(MatType data,
       const data::DatasetInfo& datasetInfo,
       LabelsType labels,
       const size_t numClasses,
       WeightsType weights,
       const size_t minimumLeafSize = 10,
       const double minimumGainSplit = 1e-7,
       const size_t maximumDepth = 0,
       DimensionSelectionType dimensionSelector = DimensionSelectionType(),
       const std::enable_if_t<arma::is_arma_type<
        typename std::remove_reference<WeightsType>::type>::value>* = 0) 
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

  //! Train on the given data, assuming all dimensions are numeric.
   /**
   * Corresponding to the public Train() method, this method is designed for
   * avoiding unnecessary copies during training.  This function is called to
   * train children.
   *
   * @param data Dataset to train on.
   * @param begin Index of the starting point in the dataset that belongs to
   *      this node.
   * @param count Number of points in this node.
   * @param datasetInfo Type information for each dimension.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @return The final entropy of decision tree.
   */
  template<bool UseWeights, typename MatType, typename WeightsType>
  double Train(MatType& data,
              const size_t begin,
              const size_t count,
              const data::DatasetInfo& datasetInfo,
              arma::Row<size_t>& labels,
              const size_t numClasses,
              WeightsType& weights,
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

    if (maximumDepth != 1)
    {
      const size_t end = dimensionSelector.End();
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
      // Store the information about the feature contributions
      featureFrequency[bestDim]++;
      featureCover[bestDim] += bestGain;

      dimensionType = (size_t) datasetInfo.Type(bestDim);
      splitDimension = bestDim;

      size_t numChildren = 0;

      // Get the number of children we will have and calculate all child assignments.
      arma::Row<size_t> childAssignments(count);
      if (datasetInfo.Type(bestDim) == data::Datatype::categorical)
      {
        // Get the number of children we will have.
        numChildren = CategoricalSplit::NumChildren(classProbabilities[0], *this);

        // Calculate all child assignments.
        for (size_t j = begin; j < begin + count; ++j)
          childAssignments[j - begin] = CategoricalSplit::CalculateDirection(
              data(bestDim, j), classProbabilities[0], *this);
      }
      else
      {
        // Get the number of children we will have.
        numChildren = NumericSplit::NumChildren(classProbabilities[0], *this);

        // Calculate all child assignments.
        for (size_t j = begin; j < begin + count; ++j)
        {
          childAssignments[j - begin] = NumericSplit::CalculateDirection(
              data(bestDim, j), classProbabilities[0], *this);
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
        XGBTree* child = new XGBTree();
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
    
    // If we didn't split, this is a leaf node.
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


  void Classify(const VecType& point,
                size_t& prediction,
                arma::vec& probabilities)
  {
    if (children.size() == 0)
    {
      prediction = majorityClass;
      probabilities = classProbabilities;
      return;
    }

    children[CalculateDirection(point)]->Classify(point, prediction,
        probabilities);
  }


  template<typename VecType>
  size_t CalculateDirection(const VecType& point) const
  {
    if ((data::Datatype) dimensionType == data::Datatype::categorical)
      return CategoricalSplit::CalculateDirection(point[splitDimension],
          classProbabilities[0], *this);
    else
      return NumericSplit::CalculateDirection(point[splitDimension],
          classProbabilities[0], *this);
  }


  private:

  //! The vector of children.
  std::vector<Node*> children;
  //! The dimension this node splits on.
  size_t splitDimension;

  union
  {
    //! Stores the type of dimension on which the split is done for internal
    //! nodes of the tree.
    size_t dimensionType;
    //! Stores the majority class for leaf nodes of the tree.
    size_t majorityClass;
  };
  /**
   * This vector may hold different things.  If the node has no children, then
   * it is guaranteed to hold the probabilities of each class.  If the node has
   * children, then it may be used arbitrarily by the split type's
   * CalculateDirection() function and may not necessarily hold class
   * probabilities.
   */
  arma::vec classProbabilities;

  //! Note that this class will also hold the members of the NumericSplit and
  //! CategoricalSplit AuxiliarySplitInfo classes, since it inherits from them.
  //! We'll define some convenience typedefs here.
  typedef typename NumericSplit::AuxiliarySplitInfo
      NumericAuxiliarySplitInfo;
  typedef typename CategoricalSplit::AuxiliarySplitInfo
      CategoricalAuxiliarySplitInfo;


};

}; // mlpack

#endif