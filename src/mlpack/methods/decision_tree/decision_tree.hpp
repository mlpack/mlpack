/**
 * @file decision_tree.hpp
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

#include <mlpack/prereqs.hpp>
#include "gini_gain.hpp"
#include "best_binary_numeric_split.hpp"
#include "all_categorical_split.hpp"

namespace mlpack {
namespace tree {

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
         typename ElemType = double,
         bool NoRecursion = false>
class DecisionTree :
    public NumericSplitType<FitnessFunction>::template
        AuxiliarySplitInfo<ElemType>,
    public CategoricalSplitType<FitnessFunction>::template
        AuxiliarySplitInfo<ElemType>
{
 public:
  //! Allow access to the numeric split type.
  typedef NumericSplitType<FitnessFunction> NumericSplit;
  //! Allow access to the categorical split type.
  typedef CategoricalSplitType<FitnessFunction> CategoricalSplit;

  /**
   * Construct the decision tree on the given data and labels, where the data
   * can be both numeric and categorical.  Setting minimumLeafSize too small may
   * cause the tree to overfit, but setting it too large may cause it to
   * underfit.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension of the dataset.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   */
  template<typename MatType>
  DecisionTree(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10);

  /**
   * Construct the decision tree on the given data and labels, assuming that the
   * data is all of the numeric type.  Setting minimumLeafSize too small may
   * cause the tree to overfit, but setting it too large may cause it to
   * underfit.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   */
  template<typename MatType>
  DecisionTree(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10);

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
   * by the datasetInfo parameter.  Setting minimumLeafSize too small may cause
   * the tree to overfit, but setting it too large may cause it to underfit.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Type information for each dimension.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   */
  template<typename MatType>
  void Train(const MatType& data,
             const data::DatasetInfo& datasetInfo,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t minimumLeafSize = 10);

  /**
   * Train the decision tree on the given data, assuming that all dimensions are
   * numeric.  This will overwrite the given model.  Setting minimumLeafSize too
   * small may cause the tree to overfit, but setting it too large may cause it
   * to underfit.
   *
   * @param data Dataset to train on.
   * @param labels Labels for each training point.
   * @param numClasses Number of classes in the dataset.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   */
  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t minimumLeafSize = 10);

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
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Get the number of children.
  size_t NumChildren() const { return children.size(); }

  //! Get the child of the given index.
  const DecisionTree& Child(const size_t i) const { return *children[i]; }
  //! Modify the child of the given index (be careful!).
  DecisionTree& Child(const size_t i) { return *children[i]; }

  /**
   * Given a point and that this node is not a leaf, calculate the index of the
   * child node this point would go towards.  This method is primarily used by
   * the Classify() function, but it can be used in a standalone sense too.
   *
   * @param point Point to classify.
   */
  template<typename VecType>
  size_t CalculateDirection(const VecType& point) const;

 private:
  //! The vector of children.
  std::vector<DecisionTree*> children;
  //! The dimension this node splits on.
  size_t splitDimension;
  //! The type of the dimension that we have split on (if we are not a leaf).
  //! If we are a leaf, then this is the index of the majority class.
  size_t dimensionTypeOrMajorityClass;
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
  typedef typename NumericSplit::template AuxiliarySplitInfo<ElemType>
      NumericAuxiliarySplitInfo;
  typedef typename CategoricalSplit::template AuxiliarySplitInfo<ElemType>
      CategoricalAuxiliarySplitInfo;

  /**
   * Calculate the class probabilities of the given labels.
   */
  template<typename RowType>
  void CalculateClassProbabilities(const RowType& labels,
                                   const size_t numClasses);
};

/**
 * Convenience typedef for decision stumps (single level decision trees).
 */
template<typename FitnessFunction = GiniGain,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit,
         typename ElemType = double>
using DecisionStump = DecisionTree<FitnessFunction,
                                   NumericSplitType,
                                   CategoricalSplitType,
                                   ElemType,
                                   false>;

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "decision_tree_impl.hpp"

#endif
