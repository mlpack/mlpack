/**
 * @file methods/decision_tree/decision_tree.hpp
 * @author Ryan Curtin
 *
 * A generic decision tree learner.  Its behavior can be controlled via template
 * arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_DECISION_TREE_HPP
#define MLPACK_METHODS_DECISION_TREE_DECISION_TREE_HPP

#include <mlpack/core.hpp>

#include "fitness_functions/fitness_functions.hpp"
#include "splits/splits.hpp"
#include "select_functions/select_functions.hpp"

namespace mlpack {

/**
 * This class implements a generic decision tree learner.  Its behavior can be
 * controlled via its template arguments.
 *
 * The class inherits from the auxiliary split information in order to prevent
 * an empty auxiliary split information struct from taking any extra size.
 */
template<typename FitnessFunction = GiniGain,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit,
         typename DimensionSelectionType = AllDimensionSelect,
         bool NoRecursion = false>
class DecisionTree :
    public NumericSplitType<FitnessFunction>::AuxiliarySplitInfo,
    public CategoricalSplitType<FitnessFunction>::AuxiliarySplitInfo
{
 public:
  //! Allow access to the numeric split type.
  using NumericSplit = NumericSplitType<FitnessFunction>;
  //! Allow access to the categorical split type.
  using CategoricalSplit = CategoricalSplitType<FitnessFunction>;
  //! Allow access to the dimension selection type.
  using DimensionSelection = DimensionSelectionType;

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
  DecisionTree(MatType data,
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
  DecisionTree(MatType data,
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
  DecisionTree(
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
          std::remove_reference_t<WeightsType>>::value>* = 0);

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
  DecisionTree(
      MatType data,
      LabelsType labels,
      const size_t numClasses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          std::remove_reference_t<WeightsType>>::value>* = 0);

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
  DecisionTree(
      const DecisionTree& other,
      MatType data,
      const data::DatasetInfo& datasetInfo,
      LabelsType labels,
      const size_t numClasses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const std::enable_if_t<arma::is_arma_type<
          std::remove_reference_t<WeightsType>>::value>* = 0);

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
  DecisionTree(
      const DecisionTree& other,
      MatType data,
      LabelsType labels,
      const size_t numClasses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Construct a decision tree without training it.  It will be a leaf node with
   * equal probabilities for each class.
   *
   * @param numClasses Number of classes in the dataset.
   */
  DecisionTree(const size_t numClasses = 1);

  /**
   * Copy another tree.  This may use a lot of memory---be sure that it's what
   * you want to do.
   *
   * @param other Tree to copy.
   */
  DecisionTree(const DecisionTree& other);

  /**
   * Take ownership of another tree.
   *
   * @param other Tree to take ownership of.
   */
  DecisionTree(DecisionTree&& other);

  /**
   * Copy another tree.  This may use a lot of memory---be sure that it's what
   * you want to do.
   *
   * @param other Tree to copy.
   */
  DecisionTree& operator=(const DecisionTree& other);

  /**
   * Take ownership of another tree.
   *
   * @param other Tree to take ownership of.
   */
  DecisionTree& operator=(DecisionTree&& other);

  /**
   * Clean up memory.
   */
  ~DecisionTree();

  /**
   * Train the decision tree on the given data.  This will overwrite the
   * existing model.  The data may have numeric and categorical types, specified
   * by the datasetInfo parameter.  Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or labels are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename LabelsType>
  double Train(MatType data,
               const data::DatasetInfo& datasetInfo,
               LabelsType labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the decision tree on the given data, assuming that all dimensions are
   * numeric.  This will overwrite the given model. Setting minimumLeafSize and
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
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename LabelsType>
  double Train(MatType data,
               LabelsType labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType());

  /**
   * Train the decision tree on the given weighted data.  This will overwrite
   * the existing model.  The data may have numeric and categorical types,
   * specified by the datasetInfo parameter.  Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights of all the labels
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  double Train(MatType data,
               const data::DatasetInfo& datasetInfo,
               LabelsType labels,
               const size_t numClasses,
               WeightsType weights,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               const std::enable_if_t<arma::is_arma_type<
                   std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Train the decision tree on the given weighted data, assuming that all
   * dimensions are numeric.  This will overwrite the given model. Setting
   * minimumLeafSize and minimumGainSplit too small may cause the tree to
   * overfit, but setting them too large may cause it to underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights of all the labels
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  double Train(MatType data,
               LabelsType labels,
               const size_t numClasses,
               WeightsType weights,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               const std::enable_if_t<arma::is_arma_type<
                   std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Classify the given point, using the entire tree.  The predicted label is
   * returned.
   *
   * @param point Point to classify.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

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
                arma::vec& probabilities) const;

  /**
   * Classify the given points, using the entire tree.  The predicted labels for
   * each point are stored in the given vector.
   *
   * @param data Set of points to classify.
   * @param predictions This will be filled with predictions for each point.
   */
  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions) const;

  /**
   * Classify the given points and also return estimates of the probabilities
   * for each class in the given matrix.  The predicted labels for each point
   * are stored in the given vector.
   *
   * @param data Set of points to classify.
   * @param predictions This will be filled with predictions for each point.
   * @param probabilities This will be filled with class probabilities for each
   *      point.
   */
  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions,
                arma::mat& probabilities) const;

  /**
   * Serialize the tree.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Get the number of children.
  size_t NumChildren() const { return children.size(); }

  //! Get the child of the given index.
  const DecisionTree& Child(const size_t i) const { return *children[i]; }
  //! Modify the child of the given index (be careful!).
  DecisionTree& Child(const size_t i) { return *children[i]; }

  //! Get the split dimension (only meaningful if this is a non-leaf in a
  //! trained tree).
  size_t SplitDimension() const { return splitDimension; }

  //! Get the class probabilities, if this is a leaf node in the trained tree.
  //! Note that if this is not a leaf, then this may contain arbitrary
  //! information used by the split in the tree!
  const arma::vec& ClassProbabilities() const { return classProbabilities; }

  /**
   * Given a point and that this node is not a leaf, calculate the index of the
   * child node this point would go towards.  This method is primarily used by
   * the Classify() function, but it can be used in a standalone sense too.
   *
   * @param point Point to classify.
   */
  template<typename VecType>
  size_t CalculateDirection(const VecType& point) const;

  /**
   * Get the number of classes in the tree.
   */
  size_t NumClasses() const;

 private:
  //! The vector of children.
  std::vector<DecisionTree*> children;
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
  using NumericAuxiliarySplitInfo = typename NumericSplit::AuxiliarySplitInfo;
  using CategoricalAuxiliarySplitInfo =
      typename CategoricalSplit::AuxiliarySplitInfo;

  /**
   * Calculate the class probabilities of the given labels.
   */
  template<bool UseWeights, typename RowType, typename WeightsRowType>
  void CalculateClassProbabilities(const RowType& labels,
                                   const size_t numClasses,
                                   const WeightsRowType& weights);

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
               DimensionSelectionType& dimensionSelector);

  /**
   * Corresponding to the public Train() method, this method is designed for
   * avoiding unnecessary copies during training.  This method is called for
   * training children.
   *
   * @param data Dataset to train on.
   * @param begin Index of the starting point in the dataset that belongs to
   *      this node.
   * @param count Number of points in this node.
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
               arma::Row<size_t>& labels,
               const size_t numClasses,
               WeightsType& weights,
               const size_t minimumLeafSize,
               const double minimumGainSplit,
               const size_t maximumDepth,
               DimensionSelectionType& dimensionSelector);
};

/**
 * Convenience typedef for decision stumps (single level decision trees).
 */
template<typename FitnessFunction = GiniGain,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit,
         typename DimensionSelectType = AllDimensionSelect>
using DecisionStump = DecisionTree<FitnessFunction,
                                   NumericSplitType,
                                   CategoricalSplitType,
                                   DimensionSelectType,
                                   false>;

/**
 * Convenience typedef for ID3 decision stumps (single level decision trees made
 * with the ID3 algorithm).
 */
using ID3DecisionStump = DecisionTree<InformationGain,
                                      BestBinaryNumericSplit,
                                      AllCategoricalSplit,
                                      AllDimensionSelect,
                                      true>;
} // namespace mlpack

// Include implementation.
#include "decision_tree_impl.hpp"

// Also include the DecisionTreeRegressor.
#include "decision_tree_regressor.hpp"

#endif
