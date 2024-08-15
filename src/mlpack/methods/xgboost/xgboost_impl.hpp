/*
 * @file methods/xgboost/xgboost_impl.hpp
 * @author Abhimanyu Dayal
 *
 * Implementation of the XGBoost class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_IMPL_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_IMPL_HPP

// Base definition of the XGBoostModel class.
#include "xgboost.hpp"
#include <mlpack/core.hpp>

// Defined within the mlpack namespace.
namespace mlpack {

void CalculateProbability(arma::mat& rawScores,
                          arma::mat& probabilities)
{  
  for (size_t i = 0; i < rawScores.n_rows; ++i)
  {
    double sumExp = 0;
    for (size_t k = 0; k < rawScores.n_cols; ++k)
      sumExp += std::exp(rawScores(i, k));
    
    for (size_t k = 0; k < rawScores.n_cols; ++k)  
      probabilities(i, k) = std::exp(rawScores(i, k)) / sumExp;
  }
}

// Empty constructor.
template<typename MatType>
XGBoost<MatType>::XGBoost() :
    numClasses(0),
    numModels(0)
{ /*Nothing to do.*/}

/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numClasses Number of classes
 * @param numModels Number of trees
 */
template<typename MatType>
XGBoost<MatType>::
  XGBoost(const MatType& data,
               const arma::Row<size_t>& labels,
               const data::DatasetInfo& datasetInfo,
               const size_t numClasses,
               const size_t numModels) :
  numClasses(numClasses),
  numModels(numModels)
{
  TrainInternal(data, labels, datasetInfo, numModels);
}

// In case the user inputs the arguments for the Weak Learner
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numClasses Number of classes
 * @param numModels Number of trees
 * @param other Weak Learner, which has been initialized already.
 */
template<typename MatType>
XGBoost<MatType>::
XGBoost(const MatType& data,
             const arma::Row<size_t>& labels,
             const data::DatasetInfo& datasetInfo,
             const size_t numClasses,
             const size_t numModels,
             const size_t minimumLeafSize, 
             const double minimumGainSplit, 
             const size_t maximumDepth) :
  numClasses(numClasses),
  numModels(numModels)
{
  TrainInternal(data, labels, numModels, 
                minimumLeafSize, minimumGainSplit, maximumDepth);
}

template<typename MatType>
void XGBoost<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const data::DatasetInfo& datasetInfo,
        const size_t numClasses,
        const size_t numModels)
{
  SetNumClasses(numClasses);
  SetNumModels(numModels);
  return TrainInternal(data, labels, datasetInfo, numModels);
}

template<typename MatType>
void XGBoost<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const data::DatasetInfo& datasetInfo,
        const size_t numClasses,
        const size_t numModels,
        const size_t minimumLeafSize, 
        const double minimumGainSplit, 
        const size_t maximumDepth)
{
  SetNumClasses(numClasses);
  SetNumModels(numModels);
  return TrainInternal(data, labels, datasetInfo, numModels,
                       minimumLeafSize, minimumGainSplit, maximumDepth);
}

// Classify the given test point.
template<typename MatType>
template<typename VecType>
size_t XGBoost<MatType>::Classify(const VecType& point) 
{
  size_t prediction;
  arma::vec probabilities; /* Not to be used */
  Classify(point, prediction, probabilities);

  return prediction;
}

template<typename MatType>
template<typename VecType>
void XGBoost<MatType>::Classify(const VecType& point,
                                size_t& prediction,
                                arma::rowvec& probabilities)
{
  arma::rowvec rawScores(numClasses, arma::fill::zeros);

  for (size_t i = 0; i < numModels; ++i) 
  {
    arma::rowvec tempRawScores = ClassifyXGBTree(point, i);
    rawScores = rawScores + tempRawScores;
  }
  CalculateProbability(rawScores, probabilities);

  prediction = 0;
  for (size_t i = 0; i < numClasses; ++i)
  {
    if (probabilities(prediction) < probabilities(i))
      prediction = i;
  }
}

template<typename MatType>
template<typename VecType>
void XGBoost<MatType>::Classify(const VecType& point,
                                size_t& prediction)
{
  VecType probabilities; /* Not to be used */
  Classify(point, prediction, probabilities);
}

template<typename MatType>
void XGBoost<MatType>::Classify(const MatType& test,
                                arma::Row<size_t>& predictedLabels,
                                arma::mat& probabilities) 
{
  predictedLabels.clear();
  predictedLabels.resize(test.n_cols);
  
  for (size_t i = 0; i < test.n_cols; ++i) 
  {
    size_t prediction;
    arma::rowvec tempProb(numClasses, arma::fill::zeros);
    arma::vec tempTest = test.col(i);

    Classify(tempTest, prediction, tempProb);
    predictedLabels(i) = prediction;

    probabilities.col(i) = tempProb;
  }
}

template<typename MatType>
void XGBoost<MatType>::Classify(const MatType& test,
                                arma::Row<size_t>& predictedLabels) 
{
  predictedLabels.clear();
  predictedLabels.resize(test.n_cols);

  arma::rowvec tempProb(numClasses, arma::fill::zeros); /* Not to be used */
  
  for (size_t i = 0; i < test.n_cols; ++i) 
  {
    size_t prediction;
    arma::vec tempTest = test.col(i);

    Classify(tempTest, prediction, tempProb);
    predictedLabels(i) = prediction;
  }
}

// TrainInternal is a private function within XGBoost class
template<typename MatType>
void XGBoost<MatType>::TrainInternal(const MatType& data,
                                     const arma::Row<size_t>& labels,
                                     const data::DatasetInfo& datasetInfo,
                                     const size_t numModels,
                                     const size_t minimumLeafSize,
                                     const double minimumGainSplit,
                                     const size_t maximumDepth) 
{

  // Clear the trees vector to in case it's preinitialised.
  trees.clear();

  // Resizing the tree vector
  trees.resize(numModels, vector<XGBTree*>(numClasses));

  // Initiate variables.
  size_t n = labels.n_elem;
  double learningRate = 1;

  // Initiate matrices for use.
  arma::mat rawScores(numClasses, n, arma::fill::zeros);
  arma::mat probabilities(numClasses, n, arma::fill::value(1.0 / n));
  arma::mat residues(numClasses, n, arma::fill::zeros);
  arma::mat tempLabels(numClasses, n, arma::fill::zeros);

  FeatureImportance* featImp = new FeatureImportance();

  // Preparing tempLabels.
  for (size_t i = 0; i < n; ++i)
    tempLabels(labels(i), i) = 1.0;

  for (size_t model = 0; model < numModels; ++model) 
  {
    // Residue is first calculated to train the weak learner.
    residues = tempLabels - probabilities;

    // The weak learner is trained.
    TrainXGBTree(data, residues, datasetInfo, model, minimumLeafSize, 
      minimumGainSplit, maximumDepth, featImp);

    // Raw scores are obtained to update probabilities variable.
    arma::mat tempPredictions = ClassifyXGBTree(data, model);

    // Raw scores are updated.
    rawScores = rawScores + learningRate * tempPredictions;

    // Probabilities are calculated.
    CalculateProbability(rawScores, probabilities);
  }
}

template<typename MatType>
void XGBoost<MatType>::TrainXGBTree(const MatType& data,
                                    const arma::mat& residue,
                                    const data::DatasetInfo& datasetInfo,
                                    const size_t modelNumber,
                                    const size_t minimumLeafSize,
                                    const double minimumGainSplit,
                                    const size_t maximumDepth,
                                    FeatureImportance* featImp)
{
  for (size_t i = 0; i < numClasses; ++i)
  {
    XGBTree* node = new XGBTree(data, residue.row(i), datasetInfo, 
      minimumLeafSize, minimumGainSplit, maximumDepth, featImp);
    trees[modelNumber][i] = node;
  }
}

template<typename MatType>
template<typename VecType>
arma::rowvec XGBoost<MatType>::ClassifyXGBTree(const VecType& data,
                                               size_t modelNumber)
{
  arma::rowvec rawScores(numClasses);

  for (size_t j = 0; j < numClasses; ++j)
    rawScores(j) = trees[modelNumber][j]->Predict(data);

  return rawScores;
}

template<typename MatType>
arma::mat XGBoost<MatType>::ClassifyXGBTree(const MatType& data,
                                            size_t modelNumber)
{
  arma::mat rawScores(numClasses, data.n_cols, arma::fill::zeros);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    for (size_t j = 0; j < numClasses; ++j)
      rawScores(j, i) = trees[modelNumber][j]->Predict(data.col(i));
  }

  return rawScores;
}

}

#endif