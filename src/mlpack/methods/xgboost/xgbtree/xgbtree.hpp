/**
 * @file methods/xgboost/xgbtree/xgbtree.hpp
 * @author Abhimanyu Dayal
 *
 * A decision tree learner specific to XGB. Its behavior can be controlled 
 * via template arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBTREE_HPP
#define MLPACK_METHODS_XGBTREE_HPP

#include <mlpack/core.hpp>
#include "node.hpp"
#include "dt_prereq.hpp"

// Defined within the mlpack namespace.
namespace mlpack {

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType,
         typename DimensionSelectionType,
         bool NoRecursion>
class XGBTree
{
  public:

  /**
   * Construct the decision tree on the given data and labels, where the data
   * can be both numeric and categorical. Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or labels are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType>
  XGBTree(MatType data,
               const data::DatasetInfo& datasetInfo,
               LabelsType labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Construct the decision tree on the given data and labels, assuming that the
   * data is all of the numeric type.  Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or labels are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType>
  XGBTree(MatType data,
               LabelsType labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Construct the decision tree on the given data and labels with weights,
   * where the data can be both numeric and categorical. Setting minimumLeafSize
   * and minimumGainSplit too small may cause the tree to overfit, but setting
   * them too large may cause it to underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  XGBTree(
      MatType data,
      const data::DatasetInfo& datasetInfo,
      LabelsType labels,
      const size_t numClasses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          typename std::remove_reference<WeightsType>::type>::value>* = 0);

  /**
   * Construct the decision tree on the given data and labels with weights,
   * assuming that the data is all of the numeric type. Setting minimumLeafSize
   * and minimumGainSplit too small may cause the tree to overfit, but setting
   * them too large may cause it to underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  XGBTree(
      MatType data,
      LabelsType labels,
      const size_t numClasses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          typename std::remove_reference<WeightsType>::type>::value>* = 0);

  /**
   * Using the hyperparameters of another decision tree, train on the given data
   * and labels with weights, where the data can be both numeric and
   * categorical.  Setting minimumLeafSize and minimumGainSplit too small may
   * cause the tree to overfit, but setting them too large may cause it to
   * underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   *
   * @param other Tree to take ownership of.
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  XGBTree(
      const XGBTree& other,
      MatType data,
      const data::DatasetInfo& datasetInfo,
      LabelsType labels,
      const size_t numClasses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const std::enable_if_t<arma::is_arma_type<
          typename std::remove_reference<WeightsType>::type>::value>* = 0);

  /**
   * Take ownership of another decision tree and train on the given data and
   * labels with weights, assuming that the data is all of the numeric type.
   * Setting minimumLeafSize and minimumGainSplit too small may cause the tree
   * to overfit, but setting them too large may cause it to underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   * @param other Tree to take ownership of.
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights The Weight list of given labels.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  XGBTree(
      const XGBTree& other,
      MatType data,
      LabelsType labels,
      const size_t numClasses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          typename std::remove_reference<WeightsType>::type>::value>* = 0);

  /**
   * Construct a decision tree without training it.  It will be a leaf node with
   * equal probabilities for each class.
   *
   * @param numClasses Number of classes in the dataset.
   */
  XGBTree(const size_t numClasses = 1);

  /**
   * Copy another tree.  This may use a lot of memory---be sure that it's what
   * you want to do.
   *
   * @param other Tree to copy.
   */
  XGBTree(const XGBTree& other);

  /**
   * Take ownership of another tree.
   *
   * @param other Tree to take ownership of.
   */
  XGBTree(XGBTree&& other);

  /**
   * Copy another tree.  This may use a lot of memory---be sure that it's what
   * you want to do.
   *
   * @param other Tree to copy.
   */
  XGBTree& operator=(const XGBTree& other);

  /**
   * Take ownership of another tree.
   *
   * @param other Tree to take ownership of.
   */
  XGBTree& operator=(XGBTree&& other);

  /**
   * Clean up memory.
   */
  ~XGBTree();

  /**
   * Prune the tree to reduce the chances of a overfitting by reducing the branches
   * with lower gain than a threshold value.
   *
   * @param threshold Determines the minimum required gain for a particular node
   * below which the node may be pruned.
   */
  bool Prune(double threshold);

  //! Get the Feature frequency information
  map<size_t, size_t> FeatureFrequency() { return featureFrequency; }

  //! Get the Feature gain information
  map<size_t, double> FeatureCover() { return featureCover; }

  /**
   * Build the tree using Node class.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
  */
  template<typename MatType, typename LabelsType, typename WeightsType>
  void Build(MatType data,
             const data::DatasetInfo& datasetInfo,
             LabelsType labels,
             const size_t numClasses,
             WeightsType weights,
             const size_t minimumLeafSize = 10,
             const double minimumGainSplit = 1e-7,
             const size_t maximumDepth = 0,
             DimensionSelectionType dimensionSelector = DimensionSelectionType(),
             const std::enable_if_t<arma::is_arma_type<
                 typename std::remove_reference<WeightsType>::type>::value>* = 0);
  
  /**
   * Classify the given point and also return estimates of the probability for
   * each class in the given vector.
   *
   * @param point Point to classify.
   * @param prediction This will be set to the predicted class of the point.
   * @param probabilities This will be filled with class probabilities for the
   *      point.
   */
  template<typename VecType>
  void Classify(const VecType& point,
                size_t& prediction,
                arma::vec& probabilities);


  private: 

  Node* root;
  //! Stores the number of times a particular feature was used to split
  map<size_t, size_t> featureFrequency;
  //! Stores the net gain provided by a particular feature for splitting
  map<size_t, double> featureCover;

  
};

}; // mlpack

#endif