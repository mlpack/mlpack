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
  using RandomForest<FitnessFunction,
                     DimensionSelectionType,
                     NumericSplitType,
                     CategoricalSplitType,
                     UseBootstrap,
                     DecisionTree<FitnessFunction,
                                  NumericSplitType,
                                  CategoricalSplitType,
                                  DimensionSelectionType>>::RandomForest;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "random_forest_classifier_impl.hpp"

#endif
