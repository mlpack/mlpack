/**
 * @file methods/random_forest/random_forest_classifier.hpp
 * @author Rishabh Garg
 *
 * Definition of the RandomForestClassifier class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_CLASSIFIER_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_CLASSIFIER_HPP

#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include "random_forest.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction = GiniGain,
         typename DimensionSelectionType = MultipleRandomDimensionSelect,
         template<typename> class NumericSplitType = BestBinaryNumericSplit,
         template<typename> class CategoricalSplitType = AllCategoricalSplit,
         bool UseBootstrap = true>
class RandomForestClassifier :
    public RandomForest<FitnessFunction,
                        DimensionSelectionType,
                        NumericSplitType,
                        CategoricalSplitType,
                        UseBootstrap,
                        DecisionTree<FitnessFunction,
                                     NumericSplitType,
                                     CategoricalSplitType,
                                     DimensionSelectionType>>
{
 public:
  /**
   * Inheriting all the constructors from the base class.
   */
  using RandomForest<FitnessFunction,
                     DimensionSelectionType,
                     NumericSplitType,
                     CategoricalSplitType,
                     UseBootstrap,
                     DecisionTree<FitnessFunction,
                                  NumericSplitType,
                                  CategoricalSplitType,
                                  DimensionSelectionType>>::RandomForest;

  /**
   * Predict the class of the given point.  If the random forest has not been
   * trained, this will throw an exception.
   *
   * @param point Point to be classified.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Predict the class of the given point and return the predicted class
   * probabilities for each class.  If the random forest has not been trained,
   * this will throw an exception.
   *
   * @param point Point to be classified.
   * @param prediction size_t to store predicted class in.
   * @param probabilities Output vector of class probabilities.
   */
  template<typename VecType>
  void Classify(const VecType& point,
                size_t& prediction,
                arma::vec& probabilities) const;

  /**
   * Predict the classes of each point in the given dataset.  If the random
   * forest has not been trained, this will throw an exception.
   *
   * @param data Dataset to be classified.
   * @param predictions Output predictions for each point in the dataset.
   */
  template<typename MatType, typename LabelsType>
  void Classify(const MatType& data,
                LabelsType& predictions) const;

  /**
   * Predict the classes of each point in the given dataset, also returning the
   * predicted class probabilities for each point.  If the random forest has not
   * been trained, this will throw an exception.
   *
   * @param data Dataset to be classified.
   * @param predictions Output predictions for each point in the dataset.
   * @param probabilities Output matrix of class probabilities for each point.
   */
  template<typename MatType, typename LabelsType>
  void Classify(const MatType& data,
                LabelsType& predictions,
                arma::mat& probabilities) const;
};

/**
 * Convenience typedef for Extra Trees. (Extremely Randomized Trees Forest)
 *
 * @code
 * @article{10.1007/s10994-006-6226-1,
 *   author = {Geurts, Pierre and Ernst, Damien and Wehenkel, Louis},
 *   title = {Extremely Randomized Trees},
 *   year = {2006},
 *   issue_date = {April 2006},
 *   publisher = {Kluwer Academic Publishers},
 *   address = {USA},
 *   volume = {63},
 *   number = {1},
 *   issn = {0885-6125},
 *   url = {https://doi.org/10.1007/s10994-006-6226-1},
 *   doi = {10.1007/s10994-006-6226-1},
 *   journal = {Mach. Learn.},
 *   month = apr,
 *   pages = {3–42},
 *   numpages = {40},
 * }
 * @endcode
 */
// template<typename FitnessFunction = GiniGain,
//          typename DimensionSelectionType = MultipleRandomDimensionSelect,
//          template<typename> class CategoricalSplitType = AllCategoricalSplit>
// using ExtraTreesClassifier = RandomForestClassifier<FitnessFunction,
//                                                     DimensionSelectionType,
//                                                     RandomBinaryNumericSplit,
//                                                     CategoricalSplitType,
//                                                     false>;

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "random_forest_classifier_impl.hpp"

#endif
