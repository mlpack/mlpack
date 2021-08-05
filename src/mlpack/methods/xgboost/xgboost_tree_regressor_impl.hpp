/**
 * @file methods/xgboost/xgboost_tree_regressor_impl.hpp
 * @author Rishabh Garg
 *
 * Implementation of XGBoost.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_TREE_REGRESSOR_IMPL_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_TREE_REGRESSOR_IMPL_HPP

// In case it hasn't been included yet.
#include "xgboost_tree_regressor.hpp"

namespace mlpack {
namespace ensemble {

template<typename LossFunction,
         typename DimensionSelectionType,
         template <typename> class NumericSplitType>
XGBoostTreeRegressor<
    LossFunction,
    DimensionSelectionType,
    NumericSplitType
>::XGBoostTreeRegressor() : eta(0.3), initialPred(0.0), avgGain(0.0)
{ /* Nothing to do here. */ }

template<typename LossFunction,
         typename DimensionSelectionType,
         template <typename> class NumericSplitType>
template<typename MatType, typename ResponsesType>
XGBoostTreeRegressor<
    LossFunction,
    DimensionSelectionType,
    NumericSplitType
>::XGBoostTreeRegressor(const MatType& dataset,
                        const ResponsesType& responses,
                        const size_t numTrees,
                        const size_t maxDepth,
                        const double eta,
                        const double minChildWeight,
                        const double gamma,
                        const double lambda,
                        const double alpha,
                        const bool warmStart,
                        DimensionSelectionType dimensionSelector) :
    eta(eta), initialPred(0.0), avgGain(0.0)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored.
  Train<false>(dataset, info, responses, numTrees, maxDepth, minChildWeight,
      gamma, lambda, alpha, warmStart, dimensionSelector);
}

template<typename LossFunction,
         typename DimensionSelectionType,
         template <typename> class NumericSplitType>
template<typename MatType, typename ResponsesType>
double XGBoostTreeRegressor<
    LossFunction,
    DimensionSelectionType,
    NumericSplitType
>::XGBoostTreeRegressor::Train(const MatType& dataset,
                               const ResponsesType& responses,
                               const size_t numTrees,
                               const size_t maxDepth,
                               const double minChildWeight,
                               const double gamma,
                               const double lambda,
                               const double alpha,
                               const bool warmStart,
                               DimensionSelectionType dimensionSelector)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored.
  return Train<false>(dataset, info, responses, numTrees, maxDepth,
      minChildWeight, gamma, lambda, alpha, warmStart, dimensionSelector);
}

template<typename LossFunction,
         typename DimensionSelectionType,
         template <typename> class NumericSplitType>
template<typename VecType>
typename VecType::elem_type XGBoostTreeRegressor<
    LossFunction,
    DimensionSelectionType,
    NumericSplitType
>::XGBoostTreeRegressor::Predict(const VecType& point) const
{
  // Check edge case.
  if (trees.size() == 0)
  {
    throw std::invalid_argument("XGBoostTreeRegressor::Predict(): "
        "xgboost is not trained yet!");
  }

  typename VecType::elem_type prediction = initialPred;

  for (size_t i = 0; i < trees.size(); ++i)
    prediction += eta * trees[i].Predict(point);

  return prediction;
}

template<typename LossFunction,
         typename DimensionSelectionType,
         template <typename> class NumericSplitType>
template<typename MatType>
void XGBoostTreeRegressor<
    LossFunction,
    DimensionSelectionType,
    NumericSplitType
>::XGBoostTreeRegressor::Predict(const MatType& data,
                                 arma::rowvec& predictions) const
{
  // Check edge case.
  if (trees.size() == 0)
  {
    predictions.clear();

    throw std::invalid_argument("XGBoostTreeRegressor::Predict(): "
        "xgboost is not trained yet!");
  }

  predictions.set_size(data.n_cols);

  #pragma omp parallel for
  for (omp_size_t i = 0; i < data.n_cols; ++i)
  {
    predictions[i] = Predict(data.col(i));
  }
}

template<typename LossFunction,
         typename DimensionSelectionType,
         template <typename> class NumericSplitType>
template<bool UseDatasetInfo, typename MatType, typename ResponsesType>
double XGBoostTreeRegressor<
    LossFunction,
    DimensionSelectionType,
    NumericSplitType
>::XGBoostTreeRegressor::Train(const MatType& dataset,
                               const data::DatasetInfo& datasetInfo,
                               const ResponsesType& responses,
                               const size_t numTrees,
                               const size_t maxDepth,
                               const double minChildWeight,
                               const double gamma,
                               const double lambda,
                               const double alpha,
                               const bool warmStart,
                               DimensionSelectionType dimensionSelector)
{
  // Reset if we are not doing a warm-start.
  if (!warmStart)
    trees.clear();
  const size_t oldNumTrees = trees.size();
  trees.resize(trees.size() + numTrees);

  // Convert avgGain to total gain.
  double totalGain = avgGain * oldNumTrees;

  // The input matrix has two rows. The first row stores the true responses and
  // the second row contains the predictions (F_m) at the current interation of
  // the boosting. It is passed to the DecisionTree in place of responses
  // because for evaluating residuals for training the tree, both the true
  // responses and the predictions at current iteration are required.
  arma::Mat<typename ResponsesType::elem_type> input(2, dataset.n_cols);
  input.row(0) = responses;

  // Calculate the value of initial prediction for boosting only when
  // warmStart is false.
  if (!warmStart)
  {
    initialPred = LossFunction::InitialPrediction(responses);
    // F_0. Set initialPred as the prediction for all the data points. This
    // serves as the basis to calculate the initial residuals on which the
    // trees are fitted. This is updated after every iteration of boosting.
    input.row(1).fill(initialPred);
  }
  // Populate the second row of input with current predictions when
  //  warmStart is true.
  else
  {
    arma::rowvec currPredictions;
    Predict(dataset, currPredictions);
    input.row(1) = currPredictions;
  }


  for (size_t i = 0; i < numTrees; ++i)
  {
    // Train each tree individually.
    Timer::Start("train_tree");
    if (UseDatasetInfo)
    {
      // minimumLeafSize parameter of the decision tree is ignored and
      // gamma is passed as the minimumGainSplit.
      totalGain += trees[oldNumTrees + i].Train(dataset, datasetInfo, input, 0,
          gamma, maxDepth, dimensionSelector,
          LossFunction(alpha, lambda, minChildWeight));
    }
    else
    {
      // minimumLeafSize parameter of the decision tree is ignored and
      // gamma is passed as the minimumGainSplit.
      totalGain += trees[oldNumTrees + i].Train(dataset, input, 0, gamma,
          maxDepth, dimensionSelector,
          LossFunction(alpha, lambda, minChildWeight));
    }
    Timer::Stop("train_tree");

    // Predict the residuals for the current step.
    arma::rowvec currResidualPred;
    trees[oldNumTrees + i].Predict(dataset, currResidualPred);

    // Update the predictions.
    // F_m(x) = F_m-1(x) + learning_rate * tree.Predict(x)
    input.row(1) += eta * currResidualPred;
  }

  avgGain = totalGain / trees.size();
  return avgGain;
}

} // namespace ensemble
} // namespace mlpack

#endif