/**
 * @file methods/decision_tree/decision_tree_regressor.hpp
 * @author Rishabh Garg
 *
 * The decision tree regressor class. Its behavior can be controlled via the
 * template arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_DECISION_TREE_REGRESSOR_HPP
#define MLPACK_METHODS_DECISION_TREE_DECISION_TREE_REGRESSOR_HPP

#include <mlpack/prereqs.hpp>
#include "mad_gain.hpp"
#include "mse_gain.hpp"
#include "best_binary_numeric_split.hpp"
#include "all_categorical_split.hpp"
#include "all_dimension_select.hpp"
#include <type_traits>


namespace mlpack {
namespace tree {

/**
 * This class implements a generic decision tree learner.  Its behavior can be
 * controlled via its template arguments.
 *
 * The class inherits from the auxiliary split information in order to prevent
 * an empty auxiliary split information struct from taking any extra size.
 */
template<typename FitnessFunction = MSEGain,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit,
         typename DimensionSelectionType = AllDimensionSelect,
         bool NoRecursion = false>
class DecisionTreeRegressor :
    public NumericSplitType<FitnessFunction>::AuxiliarySplitInfo,
    public CategoricalSplitType<FitnessFunction>::AuxiliarySplitInfo
{
 public:
  //! Allow access to the numeric split type.
  typedef NumericSplitType<FitnessFunction> NumericSplit;
  //! Allow access to the categorical split type.
  typedef CategoricalSplitType<FitnessFunction> CategoricalSplit;
  //! Allow access to the dimension selection type.
  typedef DimensionSelectionType DimensionSelection;

  /**
   * Construct a decision tree without training it.  It will be a leaf node.
   */
  DecisionTreeRegressor();

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
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType>
  DecisionTreeRegressor(MatType data,
                        const data::DatasetInfo& datasetInfo,
                        LabelsType labels,
                        const size_t minimumLeafSize = 10,
                        const double minimumGainSplit = 1e-7,
                        const size_t maximumDepth = 0,
                        DimensionSelectionType dimensionSelector =
                            DimensionSelectionType());

  /**
   * Construct the decision tree on the given data and labels, assuming that
   * the data is all of the numeric type.  Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or labels are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType>
  DecisionTreeRegressor(MatType data,
                        LabelsType labels,
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
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  DecisionTreeRegressor(
      MatType data,
      const data::DatasetInfo& datasetInfo,
      LabelsType labels,
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
   * @param weights The Weight list of given labels.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  DecisionTreeRegressor(
      MatType data,
      LabelsType labels,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          typename std::remove_reference<WeightsType>::type>::value>* = 0);

  /**
   * Take ownership of another decision tree and train on the given data and
   * labels with weights, where the data can be both numeric and categorical.
   * Setting minimumLeafSize and minimumGainSplit too small may cause the
   * tree to overfit, but setting them too large may cause it to underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   *
   * @param other Tree to take ownership of.
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param labels Labels for each training point.
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  DecisionTreeRegressor(
      const DecisionTreeRegressor& other,
      MatType data,
      const data::DatasetInfo& datasetInfo,
      LabelsType labels,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const std::enable_if_t<arma::is_arma_type<
          typename std::remove_reference<WeightsType>::type>::value>* = 0);

  /**
   * Take ownership of another decision tree and train on the given data and labels
   * with weights, assuming that the data is all of the numeric type. Setting
   * minimumLeafSize and minimumGainSplit too small may cause the tree to
   * overfit, but setting them too large may cause it to underfit.
   *
   * Use std::move if data, labels or weights are no longer needed to avoid
   * copies.
   * @param other Tree to take ownership of.
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param weights The Weight list of given labels.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename LabelsType, typename WeightsType>
  DecisionTreeRegressor(
      const DecisionTreeRegressor& other,
      MatType data,
      LabelsType labels,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          typename std::remove_reference<WeightsType>::type>::value>* = 0);

  /**
   * Copy another tree.  This may use a lot of memory---be sure that it's what
   * you want to do.
   *
   * @param other Tree to copy.
   */
  DecisionTreeRegressor(const DecisionTreeRegressor& other);

  /**
   * Take ownership of another tree.
   *
   * @param other Tree to take ownership of.
   */
  DecisionTreeRegressor(DecisionTreeRegressor&& other);

  /**
   * Copy another tree.  This may use a lot of memory---be sure that it's what
   * you want to do.
   *
   * @param other Tree to copy.
   */
  DecisionTreeRegressor& operator=(const DecisionTreeRegressor& other);

  /**
   * Take ownership of another tree.
   *
   * @param other Tree to take ownership of.
   */
  DecisionTreeRegressor& operator=(DecisionTreeRegressor&& other);

  /**
   * Clean up memory.
   */
  ~DecisionTreeRegressor();

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
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename LabelsType>
  double Train(MatType data,
               LabelsType labels,
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
               WeightsType weights,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               const std::enable_if_t<arma::is_arma_type<typename
                   std::remove_reference<WeightsType>::type>::value>* = 0);

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
               WeightsType weights,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               const std::enable_if_t<arma::is_arma_type<typename
                   std::remove_reference<WeightsType>::type>::value>* = 0);
};


} // namespace tree
} // namespace mlpack

// Include implementation.
#include "decision_tree_regressor_impl.hpp"

#endif
