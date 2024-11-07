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
  using NumericSplit = NumericSplitType<FitnessFunction>;
  //! Allow access to the categorical split type.
  using CategoricalSplit = CategoricalSplitType<FitnessFunction>;
  //! Allow access to the dimension selection type.
  using DimensionSelection = DimensionSelectionType;

  /**
   * Construct a decision tree without training it.  It will be a leaf node.
   */
  DecisionTreeRegressor();

  /**
   * Construct the decision tree on the given data and responses, where the
   * data can be both numeric and categorical. Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or responses are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param responses Responses for each training point.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename ResponsesType>
  DecisionTreeRegressor(MatType data,
                        const data::DatasetInfo& datasetInfo,
                        ResponsesType responses,
                        const size_t minimumLeafSize = 10,
                        const double minimumGainSplit = 1e-7,
                        const size_t maximumDepth = 0,
                        DimensionSelectionType dimensionSelector =
                            DimensionSelectionType());

  /**
   * Construct the decision tree on the given data and responses, assuming that
   * the data is all of the numeric type.  Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or responses are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param responses Responses for each training point.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename ResponsesType>
  DecisionTreeRegressor(MatType data,
                        ResponsesType responses,
                        const size_t minimumLeafSize = 10,
                        const double minimumGainSplit = 1e-7,
                        const size_t maximumDepth = 0,
                        DimensionSelectionType dimensionSelector =
                            DimensionSelectionType());

  /**
   * Construct the decision tree on the given data and responses with weights,
   * where the data can be both numeric and categorical. Setting minimumLeafSize
   * and minimumGainSplit too small may cause the tree to overfit, but setting
   * them too large may cause it to underfit.
   *
   * Use std::move if data, responses or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param responses Responses for each training point.
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename ResponsesType, typename WeightsType>
  DecisionTreeRegressor(
      MatType data,
      const data::DatasetInfo& datasetInfo,
      ResponsesType responses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Construct the decision tree on the given data and responses with weights,
   * assuming that the data is all of the numeric type. Setting minimumLeafSize
   * and minimumGainSplit too small may cause the tree to overfit, but setting
   * them too large may cause it to underfit.
   *
   * Use std::move if data, responses or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param responses Responses for each training point.
   * @param weights The Weight list of given labels.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename ResponsesType, typename WeightsType>
  DecisionTreeRegressor(
      MatType data,
      ResponsesType responses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Take ownership of another decision tree and train on the given data and
   * responses with weights, where the data can be both numeric and
   * categorical. Setting minimumLeafSize and minimumGainSplit too small may
   * cause the tree to overfit, but setting them too large may cause it to
   * underfit.
   *
   * Use std::move if data, responses or weights are no longer needed to avoid
   * copies.
   *
   * @param other Tree to take ownership of.
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param responses Responses for each training point.
   * @param weights The weight list of given label.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   */
  template<typename MatType, typename ResponsesType, typename WeightsType>
  DecisionTreeRegressor(
      const DecisionTreeRegressor& other,
      MatType data,
      const data::DatasetInfo& datasetInfo,
      ResponsesType responses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const std::enable_if_t<arma::is_arma_type<
          std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Take ownership of another decision tree and train on the given data and
   * responses with weights, assuming that the data is all of the numeric type.
   * Setting minimumLeafSize and minimumGainSplit too small may cause the tree
   * to overfit, but setting them too large may cause it to underfit.
   *
   * Use std::move if data, responses or weights are no longer needed to avoid
   * copies.
   * @param other Tree to take ownership of.
   * @param data Dataset to train on.
   * @param responses Responses for each training point.
   * @param weights The Weight list of given labels.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   */
  template<typename MatType, typename ResponsesType, typename WeightsType>
  DecisionTreeRegressor(
      const DecisionTreeRegressor& other,
      MatType data,
      ResponsesType responses,
      WeightsType weights,
      const size_t minimumLeafSize = 10,
      const double minimumGainSplit = 1e-7,
      const size_t maximumDepth = 0,
      DimensionSelectionType dimensionSelector = DimensionSelectionType(),
      const std::enable_if_t<arma::is_arma_type<
          std::remove_reference_t<WeightsType>>::value>* = 0);

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
   * existing model. The data may have numeric and categorical types, specified
   * by the datasetInfo parameter.  Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or responses are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension.
   * @param responses Responses for each training point.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @param fitnessFunction Instantiated fitnessFunction. It is used to
   *      evaluate the fitness score for splitting each node.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename ResponsesType>
  double Train(MatType data,
               const data::DatasetInfo& datasetInfo,
               ResponsesType responses,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               FitnessFunction fitnessFunction = FitnessFunction());

  /**
   * Train the decision tree on the given data, assuming that all dimensions are
   * numeric.  This will overwrite the given model. Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data or responses are no longer needed to avoid copies.
   *
   * @param data Dataset to train on.
   * @param responses Responses for each training point.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @param fitnessFunction Instantiated fitnessFunction. It is used to
   *      evaluate the fitness score for splitting each node.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename ResponsesType>
  double Train(MatType data,
               ResponsesType responses,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               FitnessFunction fitnessFunction = FitnessFunction());

  /**
   * Train the decision tree on the given weighted data.  This will overwrite
   * the existing model.  The data may have numeric and categorical types,
   * specified by the datasetInfo parameter.  Setting minimumLeafSize and
   * minimumGainSplit too small may cause the tree to overfit, but setting them
   * too large may cause it to underfit.
   *
   * Use std::move if data, responses or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension.
   * @param responses Responses for each training point.
   * @param weights Weights of all the labels
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @param fitnessFunction Instantiated fitnessFunction. It is used to
   *      evaluate the fitness score for splitting each node.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename ResponsesType, typename WeightsType>
  double Train(MatType data,
               const data::DatasetInfo& datasetInfo,
               ResponsesType responses,
               WeightsType weights,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               FitnessFunction fitnessFunction = FitnessFunction(),
               const std::enable_if_t<arma::is_arma_type<
                   std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Train the decision tree on the given weighted data, assuming that all
   * dimensions are numeric.  This will overwrite the given model. Setting
   * minimumLeafSize and minimumGainSplit too small may cause the tree to
   * overfit, but setting them too large may cause it to underfit.
   *
   * Use std::move if data, responses or weights are no longer needed to avoid
   * copies.
   *
   * @param data Dataset to train on.
   * @param responses Responses for each training point.
   * @param weights Weights of all the labels
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param dimensionSelector Instantiated dimension selection policy.
   * @param fitnessFunction Instantiated fitnessFunction. It is used to
   *      evaluate the fitness score for splitting each node.
   * @return The final entropy of decision tree.
   */
  template<typename MatType, typename ResponsesType, typename WeightsType>
  double Train(MatType data,
               ResponsesType responses,
               WeightsType weights,
               const size_t minimumLeafSize = 10,
               const double minimumGainSplit = 1e-7,
               const size_t maximumDepth = 0,
               DimensionSelectionType dimensionSelector =
                   DimensionSelectionType(),
               FitnessFunction fitnessFunction = FitnessFunction(),
               const std::enable_if_t<arma::is_arma_type<
                   std::remove_reference_t<WeightsType>>::value>* = 0);

  /**
   * Make prediction for the given point, using the entire tree.  The predicted
   * label is returned.
   *
   * @param point Point to predict.
   */
  template<typename VecType>
  typename VecType::elem_type Predict(const VecType& point) const;

  /**
   * Make prediction for the given points, using the entire tree. The predicted
   * responses for each point are stored in the given vector.
   *
   * @param data Set of points to predict.
   * @param predictions This will be filled with predictions for each point.
   */
  template<typename MatType, typename PredVecType>
  void Predict(const MatType& data,
               PredVecType& predictions) const;

  /**
   * Serialize the tree.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Get the number of children.
  size_t NumChildren() const { return children.size(); }

  //! Get the number of leaves in the tree.
  size_t NumLeaves() const;

  //! Get the child of the given index.
  const DecisionTreeRegressor& Child(const size_t i) const
  {
    return *children[i];
  }
  //! Modify the child of the given index (be careful!).
  DecisionTreeRegressor& Child(const size_t i) { return *children[i]; }

  //! Get the split dimension (only meaningful if this is a non-leaf in a
  //! trained tree).
  size_t SplitDimension() const { return splitDimension; }

  /**
   * Given a point and that this node is not a leaf, calculate the index of the
   * child node this point would go towards.  This method is primarily used by
   * the Predict() function, but it can be used in a standalone sense too.
   *
   * @param point Point to predict.
   */
  template<typename VecType>
  size_t CalculateDirection(const VecType& point) const;

 private:
  //! The vector of children.
  std::vector<DecisionTreeRegressor*> children;
  union
  {
    //! Stores the prediction value, for leaf nodes of the tree.
    double prediction;
    //! The dimension of the split, for internal nodes.
    size_t splitDimension;
  };
  //! For internal nodes, the type of the split variable.
  size_t dimensionType;
  //! For internal nodes, the split information for the splitter.
  arma::vec splitInfo;

  //! Note that this class will also hold the members of the NumericSplit and
  //! CategoricalSplit AuxiliarySplitInfo classes, since it inherits from them.
  //! We'll define some convenience typedefs here.
  using NumericAuxiliarySplitInfo = typename NumericSplit::AuxiliarySplitInfo;
  using CategoricalAuxiliarySplitInfo =
      typename CategoricalSplit::AuxiliarySplitInfo;

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
   * @param responses Responses for each training point.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param fitnessFunction Instantiated fitnessFunction. It is used to
   *      evaluate the fitness score for splitting each node.
   * @return The final entropy of decision tree.
   */
  template<bool UseWeights, typename MatType, typename ResponsesType>
  double Train(MatType& data,
               const size_t begin,
               const size_t count,
               const data::DatasetInfo& datasetInfo,
               ResponsesType& responses,
               arma::rowvec& weights,
               const size_t minimumLeafSize,
               const double minimumGainSplit,
               const size_t maximumDepth,
               DimensionSelectionType& dimensionSelector,
               FitnessFunction fitnessFunction = FitnessFunction());

  /**
   * Corresponding to the public Train() method, this method is designed for
   * avoiding unnecessary copies during training.  This method is called for
   * training children.
   *
   * @param data Dataset to train on.
   * @param begin Index of the starting point in the dataset that belongs to
   *      this node.
   * @param count Number of points in this node.
   * @param responses Responses for each training point.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @param minimumGainSplit Minimum gain for the node to split.
   * @param maximumDepth Maximum depth for the tree.
   * @param fitnessFunction Instantiated fitnessFunction. It is used to
   *      evaluate the fitness score for splitting each node.
   * @return The final entropy of decision tree.
   */
  template<bool UseWeights, typename MatType, typename ResponsesType>
  double Train(MatType& data,
               const size_t begin,
               const size_t count,
               ResponsesType& responses,
               arma::rowvec& weights,
               const size_t minimumLeafSize,
               const double minimumGainSplit,
               const size_t maximumDepth,
               DimensionSelectionType& dimensionSelector,
               FitnessFunction fitnessFunction = FitnessFunction());
};


} // namespace mlpack

// Include implementation.
#include "decision_tree_regressor_impl.hpp"

#endif
