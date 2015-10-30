/**
 * @file hoeffding_split.hpp
 * @author Ryan Curtin
 *
 * An implementation of the standard Hoeffding tree by Pedro Domingos and Geoff
 * Hulten in ``Mining High-Speed Data Streams''.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_HPP

#include <mlpack/core.hpp>
#include "gini_impurity.hpp"
#include "hoeffding_numeric_split.hpp"
#include "hoeffding_categorical_split.hpp"

namespace mlpack {
namespace tree {

/**
 * The HoeffdingTree object represents all of the necessary information for a
 * Hoeffding-bound-based decision tree.  This class is able to train on samples
 * in streaming settings and batch settings, and perform splits based on the
 * Hoeffding bound.  The Hoeffding tree (also known as the "very fast decision
 * tree" -- VFDT) is described in the following paper:
 *
 * @code
 * @inproceedings{domingos2000mining,
 *     title={{Mining High-Speed Data Streams}},
 *     author={Domingos, P. and Hulten, G.},
 *     year={2000},
 *     booktitle={Proceedings of the Sixth ACM SIGKDD International Conference
 *         on Knowledge Discovery and Data Mining (KDD '00)},
 *     pages={71--80}
 * }
 * @endcode
 *
 * The class is modular, and takes three template parameters.  The first,
 * FitnessFunction, is the fitness function that should be used to determine
 * whether a split is beneficial; examples might be GiniImpurity or
 * InformationGain.  The NumericSplitType determines how numeric attributes are
 * handled, and the CategoricalSplitType determines how categorical attributes
 * are handled.  As far as the actual splitting goes, the meat of the splitting
 * procedure will be contained in those two classes.
 *
 * @tparam FitnessFunction Fitness function to use.
 * @tparam NumericSplitType Technique for splitting numeric features.
 * @tparam CategoricalSplitType Technique for splitting categorical features.
 */
template<typename FitnessFunction = GiniImpurity,
         template<typename> class NumericSplitType =
             HoeffdingDoubleNumericSplit,
         template<typename> class CategoricalSplitType =
             HoeffdingCategoricalSplit
>
class HoeffdingTree
{
 public:
  /**
   * Construct the Hoeffding tree with the given parameters and given training
   * data.  The tree may be trained either in batch mode (which looks at all
   * points before splitting, and propagates these points to the created
   * children for further training), or in streaming mode, where each point is
   * only considered once.  (In general, batch mode will give better-performing
   * trees, but will have higher memory and runtime costs for the same dataset.)
   *
   * @param data Dataset to train on.
   * @param datasetInfo Information on the dataset (types of each feature).
   * @param numClasses Number of classes in the dataset.
   * @param batchTraining Whether or not to train in batch.
   * @param successProbability Probability of success required in Hoeffding
   *      bounds before a split can happen.
   * @param maxSamples Maximum number of samples before a split is forced (0
   *      never forces a split); ignored in batch training mode.
   * @param checkInterval Number of samples required before each split; ignored
   *      in batch training mode.
   */
  template<typename MatType>
  HoeffdingTree(const MatType& data,
                const arma::Col<size_t>& labels,
                const data::DatasetInfo& datasetInfo,
                const size_t numClasses,
                const bool batchTraining = true,
                const double successProbability = 0.95,
                const size_t maxSamples = 0,
                const size_t checkInterval = 100);

  /**
   * Construct the Hoeffding tree with the given parameters, but training on no
   * data.  The dimensionMappings parameter is only used if it is desired that
   * this node does not create its own dimensionMappings object (for instance,
   * if this is a child of another node in the tree).
   *
   * @param dimensionality Dimensionality of the dataset.
   * @param numClasses Number of classes in the dataset.
   * @param datasetInfo Information on the dataset (types of each feature).
   * @param successProbability Probability of success required in Hoeffding
   *      bound before a split can happen.
   * @param maxSamples Maximum number of samples before a split is forced.
   * @param checkInterval Number of samples required before each split check.
   * @param dimensionMappings Mappings from dimension indices to positions in
   *      numeric and categorical split vectors.  If left NULL, a new one will
   *      be created.
   */
  HoeffdingTree(const data::DatasetInfo& datasetInfo,
                const size_t numClasses,
                const double successProbability,
                const size_t maxSamples,
                const size_t checkInterval,
                std::unordered_map<size_t, std::pair<size_t, size_t>>*
                    dimensionMappings = NULL);

  /**
   * Clean up memory.
   */
  ~HoeffdingTree();

  /**
   * Train on a set of points, either in streaming mode or in batch mode, with
   * the given labels.
   *
   * @param data Data points to train on.
   * @param label Labels of data points.
   * @param batchTraining If true, perform training in batch.
   */
  template<typename MatType>
  void Train(const MatType& data,
             const arma::Col<size_t>& labels,
             const bool batchTraining = true);

  /**
   * Train on a single point in streaming mode, with the given label.
   *
   * @param point Point to train on.
   * @param label Label of point to train on.
   */
  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  /**
   * Check if a split would satisfy the conditions of the Hoeffding bound with
   * the node's specified success probability.  If so, the number of children
   * that would be created is returned.  If not, 0 is returned.
   */
  size_t SplitCheck();

  //! Get the splitting dimension (size_t(-1) if no split).
  size_t SplitDimension() const { return splitDimension; }

  //! Get the majority class.
  size_t MajorityClass() const;
  //! Modify the majority class.
  size_t& MajorityClass();

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
   * Classify the given point, using this node and the entire (sub)tree beneath
   * it.  The predicted label is returned.
   *
   * @param point Point to classify.
   * @return Predicted label of point.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Classify the given point and also return an estimate of the probability
   * that the prediction is correct.  (This estimate is simply the probability
   * that a training point was from the majority class in the leaf that this
   * point binned to.)
   *
   * @param point Point to classify.
   * @param prediction Predicted label of point.
   * @param probability An estimate of the probability that the prediction is
   *      correct.
   */
  template<typename VecType>
  void Classify(const VecType& point, size_t& prediction, double& probability)
      const;

  /**
   * Given that this node should split, create the children.
   */
  template<typename StreamingDecisionTreeType>
  void CreateChildren(std::vector<StreamingDecisionTreeType>& children);

  //! Serialize the split.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  // We need to keep some information for before we have split.

  //! Information for splitting of numeric features (used before split).
  std::vector<NumericSplitType<FitnessFunction>> numericSplits;
  //! Information for splitting of categorical features (used before split).
  std::vector<CategoricalSplitType<FitnessFunction>> categoricalSplits;

  //! This structure is owned by this node only if it is the root of the tree.
  std::unordered_map<size_t, std::pair<size_t, size_t>>* dimensionMappings;
  //! Indicates whether or not we own the mappings.
  bool ownsMappings;

  //! The number of samples seen so far by this node.
  size_t numSamples;
  //! The number of classes this node is trained on.
  size_t numClasses;
  //! The maximum number of samples we can see before splitting.
  size_t maxSamples;
  //! The number of samples that should be seen before checking for a split.
  size_t checkInterval;
  //! The dataset information.  (We don't own this.)
  const data::DatasetInfo* datasetInfo;
  //! The required probability of success for a split to be performed.
  double successProbability;

  // And we need to keep some information for after we have split.

  //! The dimension that this node has split on.
  size_t splitDimension;
  //! The majority class of this node.
  size_t majorityClass;
  //! The empirical probability of a point this node saw having the majority
  //! class.
  double majorityProbability;
  //! If the split is categorical, this holds the splitting information.
  typename CategoricalSplitType<FitnessFunction>::SplitInfo categoricalSplit;
  //! If the split is numeric, this holds the splitting information.
  typename NumericSplitType<FitnessFunction>::SplitInfo numericSplit;
};

} // namespace tree
} // namespace mlpack

#include "hoeffding_tree_impl.hpp"

#endif
