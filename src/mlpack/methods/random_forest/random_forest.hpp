/**
 * @file random_forest.hpp
 * @author Ryan Curtin
 *
 * Definition of the RandomForest class.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_HPP

#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include "bootstrap.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction = GiniGain,
         typename DimensionSelectionType = MultipleRandomDimensionSelect<>,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit,
         typename ElemType = double>
class RandomForest
{
 public:
  //! Allow access to the underlying decision tree type.
  typedef DecisionTree<FitnessFunction, NumericSplitType, CategoricalSplitType,
      DimensionSelectionType, ElemType> DecisionTreeType;

  RandomForest() { }

  template<typename MatType>
  RandomForest(const MatType& dataset,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numTrees = 50,
               const size_t minimumLeafSize = 20);

  /**
   * Construct the random forest on the given dataset with the given labels.
   */
  template<typename MatType>
  RandomForest(const MatType& dataset,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numTrees = 50,
               const size_t minimumLeafSize = 20);

  template<typename MatType>
  RandomForest(const MatType& dataset,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const arma::rowvec& weights,
               const size_t numTrees = 50,
               const size_t minimumLeafSize = 20);

  template<typename MatType>
  RandomForest(const MatType& dataset,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const arma::rowvec& weights,
               const size_t numTrees = 50,
               const size_t minimumLeafSize = 20);

  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t numTrees = 50,
             const size_t minimumLeafSize = 20);

  template<typename MatType>
  void Train(const MatType& data,
             const data::DatasetInfo& datasetInfo,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t numTrees = 50,
             const size_t minimumLeafSize = 20);

  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const arma::rowvec& weights,
             const size_t numTrees = 50,
             const size_t minimumLeafSize = 20);

  template<typename MatType>
  void Train(const MatType& data,
             const data::DatasetInfo& datasetInfo,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const arma::rowvec& weights,
             const size_t numTrees = 50,
             const size_t minimumLeafSize = 20);

  template<typename VecType>
  size_t Classify(const VecType& point) const;

  template<typename VecType>
  void Classify(const VecType& point,
                size_t& prediction,
                arma::vec& probabilities) const;

  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions) const;

  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions,
                arma::mat& probabilities) const;

  DecisionTreeType& Tree(const size_t i) { return trees[i]; }
  const DecisionTreeType& Tree(const size_t i) const { return trees[i]; }

  size_t NumTrees() const { return trees.size(); }

 private:
  /**
   * Perform the training of the decision tree.  The template bool parameters
   * control whether or not the datasetInfo or weights arguments should be
   * ignored.
   *
   * @param data Dataset to train on.
   * @param datasetInfo Dimension information for the dataset (may be ignored).
   * @param labels Labels for the dataset.
   * @param numClasses Number of classes in the dataset.
   * @param weights Weights for each point in the dataset (may be ignored).
   * @param numTrees Number of trees in the forest.
   * @param minimumLeafSize Minimum number of points in each leaf node.
   * @tparam UseWeights Whether or not to use the weights parameter.
   * @tparam UseDatasetInfo Whether or not to use the datasetInfo parameter.
   * @tparam MatType The type of data matrix (i.e. arma::mat).
   */
  template<bool UseWeights, bool UseDatasetInfo, typename MatType>
  void Train(const MatType& data,
             const data::DatasetInfo& datasetInfo,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const arma::rowvec& weights,
             const size_t numTrees,
             const size_t minimumLeafSize);

  std::vector<DecisionTreeType> trees;
};

} // namespace rf
} // namespace mlpack

// Include implementation.
#include "random_forest_impl.hpp"

#endif
