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
#include "loss_functions/sse_loss.hpp"
#include <mlpack/core.hpp>

// Defined within the mlpack namespace.
namespace mlpack {

// Define a function to compute the learning rate based on the current model predictions.
template<typename MatType>
double ComputeLearningRate(const MatType& predictions,
                           const arma::Row<size_t>& labels) 
{
    // Define the loss function you are using.
    // For example, mean squared error for regression.
    // You may need to adjust this based on your problem.
    double loss = arma::mean(arma::square(predictions - labels));

    // Define a range of learning rates to search over.
    const double minLearningRate = 0.001;  // Minimum learning rate.
    const double maxLearningRate = 1.0;    // Maximum learning rate.
    const size_t numLearningRates = 100;   // Number of learning rates to try.

    // Initialize the best learning rate and the corresponding loss.
    double bestLearningRate = minLearningRate;
    double bestLoss = std::numeric_limits<double>::max();

    // Try different learning rates within the specified range.
    for (size_t i = 0; i < numLearningRates; ++i) 
    {
        double learningRate = minLearningRate +
                              (maxLearningRate - minLearningRate) *
                              static_cast<double>(i) / static_cast<double>(numLearningRates);

        // Compute the loss with the current learning rate.
        double currentLoss = arma::mean(arma::square(predictions - labels - learningRate * predictions));

        // Update the best learning rate if the current loss is lower.
        if (currentLoss < bestLoss) 
        {
            bestLearningRate = learningRate;
            bestLoss = currentLoss;
        }
    }

    return bestLearningRate;
}

arma::Row<size_t> Loss(arma::Row<size_t>& residue, arma::Row<size_t>& predictions, double& loss)
{
  // Defining alpha and lambda values (for regularization).
  double alpha = 10;
  double lambda = 10;
  
  SSELoss lossObj(alpha, lambda);

  arma::mat combinedInput(2, residue.n_cols);

  for (size_t i = 0; i < residue.n_cols ; ++i)
  {
    combinedInput(0, i) = residue(i);
    combinedInput(1, i) = predictions(i);
  }

  // Weight vector we will not use for now
  arma::mat weights; 

  loss = lossObj.Evaluate<false, arma::mat, arma::mat>(combinedInput, weights);

  return residue - predictions;
}

template<typename WeakLearnerType, typename MatType>
template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>
void XGBoost<WeakLearnerType, MatType>:: 
  TrainInternal(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numModels,
                const size_t numClasses,
                const WeakLearnerType& wl,
                WeakLearnerArgs&&... weakLearnerArgs) 
{

  // Load the initial weights into a 2-D matrix.
  const ElemType initWeight = 1.0 / ElemType(data.n_cols * numClasses);
  MatType D(numClasses, data.n_cols);
  D.fill(initWeight);

  // Weights are stored in this row vector.
  arma::Row<ElemType> weights(labels.n_cols);

  weakLearners.clear();

  // Define pruning threshold
  double pruningThreshold = 1;
  
  arma::Row<size_t> residue = labels;

  for (size_t model = 0; model < numModels; ++model) 
  {

    // Build the weight vectors.
    weights = sum(D);

    WeakLearnerType* wPtr;

    if(UseExistingWeakLearner)
    {
      wPtr = new WeakLearnerType(wl, data, residue, numClasses, weights,
        std::forward<WeakLearnerArgs>(weakLearnerArgs)...);
    }
    else 
    {
      wPtr = new WeakLearnerType(data, residue, numClasses,
        std::forward<WeakLearnerArgs>(weakLearnerArgs)...);
    }

    WeakLearnerType w = *wPtr;

    arma::Row<size_t> predictions(residue.n_cols, arma::fill::zeros);
    w.Classify(data, predictions);
    w.Prune(pruningThreshold);

    weakLearners.push_back(w);

    // Compute the learning rate for this iteration.
    double learningRate = ComputeLearningRate(predictions, residue);

    predictions = learningRate * predictions;

    double loss; 
    residue = Loss(residue, predictions, loss);

    delete wPtr;
        
  }
}

}

#endif