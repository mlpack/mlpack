/**
 * @file methods/random_forest/random_forest_classifier_impl.hpp
 * @author Rishabh Garg
 *
 * Implementation of the RandomForestClassifier class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_CLASSIFIER_IMPL_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_CLASSIFIER_IMPL_HPP

// In case it hasn't been included yet.
#include "random_forest_classifier.hpp"

namespace mlpack {
namespace tree {

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename VecType>
size_t RandomForestClassifier<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Classify(const VecType& point) const
{
  // Pass off to another Classify() overload.
  size_t predictedClass;
  arma::vec probabilities;
  Classify(point, predictedClass, probabilities);

  return predictedClass;
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename VecType>
void RandomForestClassifier<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Classify(const VecType& point,
            size_t& prediction,
            arma::vec& probabilities) const
{
  // Check edge case.
  if (this->trees.size() == 0)
  {
    probabilities.clear();
    prediction = 0;

    throw std::invalid_argument("RandomForest::Classify(): no random forest "
        "trained!");
  }

  probabilities.zeros(this->trees[0].NumClasses());
  for (size_t i = 0; i < this->trees.size(); ++i)
  {
    arma::vec treeProbs;
    size_t treePrediction; // Ignored.
    this->trees[i].Classify(point, treePrediction, treeProbs);

    probabilities += treeProbs;
  }

  // Find maximum element after renormalizing probabilities.
  probabilities /= this->trees.size();
  arma::uword maxIndex = 0;
  probabilities.max(maxIndex);

  // Set prediction.
  prediction = (size_t) maxIndex;
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType, typename LabelsType>
void RandomForestClassifier<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Classify(const MatType& data,
            LabelsType& predictions) const
{
  // Check edge case.
  if (this->trees.size() == 0)
  {
    predictions.clear();

    throw std::invalid_argument("RandomForest::Classify(): no random forest "
        "trained!");
  }

  predictions.set_size(data.n_cols);

  #pragma omp parallel for
  for (omp_size_t i = 0; i < data.n_cols; ++i)
  {
    predictions[i] = Classify(data.col(i));
  }
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType, typename LabelsType>
void RandomForestClassifier<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Classify(const MatType& data,
            LabelsType& predictions,
            arma::mat& probabilities) const
{
  // Check edge case.
  if (this->trees.size() == 0)
  {
    predictions.clear();
    probabilities.clear();

    throw std::invalid_argument("RandomForestClassifier::Classify(): no random"
        " forest trained!");
  }

  probabilities.set_size(this->trees[0].NumClasses(), data.n_cols);
  predictions.set_size(data.n_cols);
  #pragma omp parallel for
  for (omp_size_t i = 0; i < data.n_cols; ++i)
  {
    arma::vec probs = probabilities.unsafe_col(i);
    Classify(data.col(i), predictions[i], probs);
  }
}

} // namespace tree
} // namespace mlpack

#endif
