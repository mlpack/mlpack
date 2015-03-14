/*
 * @file adaboost_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of the AdaBoost class.
 *
 * @code
 * @article{schapire1999improved,
 *   author = {Schapire, Robert E. and Singer, Yoram},
 *   title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
 *   journal = {Machine Learning},
 *   volume = {37},
 *   number = {3},
 *   month = dec,
 *   year = {1999},
 *   issn = {0885-6125},
 *   pages = {297--336},
 * }
 * @endcode
 */
#ifndef __MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP
#define __MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP

#include "adaboost.hpp"

namespace mlpack {
namespace adaboost {

/**
 *  Constructor. Currently runs the AdaBoost.mh algorithm
 *
 *  @param data Input data
 *  @param labels Corresponding labels
 *  @param iterations Number of boosting rounds
 *  @param other Weak Learner, which has been initialized already
 */
template<typename MatType, typename WeakLearner>
AdaBoost<MatType, WeakLearner>::AdaBoost(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const int iterations,
    const double tol,
    const WeakLearner& other)
{
  // Count the number of classes.
  numClasses = (arma::max(labels) - arma::min(labels)) + 1;
  tolerance = tol;

  double rt, crt, alphat = 0.0, zt;

  // crt is for stopping the iterations when rt
  // stops changing by less than a tolerant value.

  // crt is cumulative rt for stopping the iterations when rt
  // stops changing by less than a tolerant value.

  ztProduct = 1.0;

  // To be used for prediction by the Weak Learner for prediction.
  arma::Row<size_t> predictedLabels(labels.n_cols);

  // Use tempData to modify input Data for incorporating weights.
  MatType tempData(data);

  // This matrix is a helper matrix used to calculate the final hypothesis.
  arma::mat sumFinalH(predictedLabels.n_cols, numClasses);
  sumFinalH.fill(0.0);

  // load the initial weights into a 2-D matrix
  const double initWeight = 1.0 / double(data.n_cols * numClasses);
  arma::mat D(data.n_cols, numClasses);
  D.fill(initWeight);

  // Weights are to be compressed into this rowvector
  // for focussing on the perceptron weights.
  arma::rowvec weights(predictedLabels.n_cols);

  // This is the final hypothesis.
  arma::Row<size_t> finalH(predictedLabels.n_cols);

  // now start the boosting rounds
  for (int i = 0; i < iterations; i++)
  {
    // Initialized to zero in every round.
    // rt is used for calculation of alphat, is the weighted error
    // rt = (sum)D(i)y(i)ht(xi)
    rt = 0.0;

    // zt is used for weight normalization.
    zt = 0.0;

    // Build the weight vectors
    BuildWeightMatrix(D, weights);

    // call the other weak learner and train the labels.
    WeakLearner w(other, tempData, weights, labels);
    w.Classify(tempData, predictedLabels);

    // Now from predictedLabels, build ht, the weak hypothesis
    // buildClassificationMatrix(ht, predictedLabels);

    // Now, start calculation of alpha(t) using ht

    for (size_t j = 0;j < D.n_rows; j++) // instead of D, ht
    {
      if (predictedLabels(j) == labels(j))
      {
        // for (int k = 0;k < numClasses; k++)
        //   rt += D(j,k);
        rt += arma::accu(D.row(j));
      }

      else
      {
        // for (int k = 0;k < numClasses; k++)
        //   rt -= D(j,k);
        rt -= arma::accu(D.row(j));
      }
    }
    // end calculation of rt

    if (i > 0)
    {
      if (std::abs(rt - crt) < tolerance)
        break;
    }
    crt = rt;

    // our goal is to find alphat which mizimizes or approximately minimizes
    // the value of Z as a function of alpha.
    alphat = 0.5 * log((1 + rt) / (1 - rt));
    // end calculation of alphat

    alpha.push_back(alphat);
    wl.push_back(w);

    // now start modifying weights
    for (size_t j = 0;j < D.n_rows; j++)
    {
      double expo = exp(alphat);
      if (predictedLabels(j) == labels(j))
      {
          for (size_t k = 0;k < D.n_cols; k++)
          {
            // we calculate zt, the normalization constant
            zt += D(j,k) / expo; // * exp(-1 * alphat * yt(j,k) * ht(j,k));
            D(j,k) = D(j,k) / expo;

            // adding to the matrix of FinalHypothesis
            // sumFinalH(j,k) += (alphat * ht(j,k));
            if (k == labels(j))
              sumFinalH(j,k) += (alphat);// * ht(j,k));
            else
              sumFinalH(j,k) -= (alphat);
          }
      }
      else
      {
        for (size_t k = 0;k < D.n_cols; k++)
          {
            // we calculate zt, the normalization constant
            zt += D(j,k) * expo;
            D(j,k) = D(j,k) * expo;

            // adding to the matrix of FinalHypothesis
            if (k == labels(j))
              sumFinalH(j,k) += (alphat);// * ht(j,k));
            else
              sumFinalH(j,k) -= (alphat);
          }
      }
    }

    // normalization of D
    D = D / zt;

    // Accumulating the value of zt for the Hamming Loss bound.
    ztProduct *= zt;
  }

  // Iterations are over, now build a strong hypothesis
  // from a weighted combination of these weak hypotheses.

  arma::colvec tempSumFinalH;
  arma::uword max_index;
  arma::mat sfh = sumFinalH.t();

  for (size_t i = 0;i < sfh.n_cols; i++)
  {
    tempSumFinalH = sfh.col(i);
    tempSumFinalH.max(max_index);
    finalH(i) = max_index;
  }
  finalHypothesis = finalH;
}

/**
 * Classification Function.
 * @param test Testing data.
 * @param predictedLabels Vector to store the predicted labels of the
 *                         test set.
 */
template <typename MatType, typename WeakLearner>
void AdaBoost<MatType, WeakLearner>::Classify(
    const MatType& test,
    arma::Row<size_t>& predictedLabels)
{
  arma::Row<size_t> tempPredictedLabels(predictedLabels.n_cols);
  arma::mat cMatrix(numClasses, test.n_cols);

  cMatrix.zeros();
  predictedLabels.zeros();

  for (size_t i = 0;i < wl.size(); i++)
  {
    wl[i].Classify(test, tempPredictedLabels);

    for (size_t j = 0; j < tempPredictedLabels.n_cols; j++)
      cMatrix(tempPredictedLabels(j), j) += (alpha[i] * tempPredictedLabels(j));
  }

  arma::colvec cMRow;
  arma::uword max_index;

  for (size_t i = 0; i < predictedLabels.n_cols; i++)
  {
    cMRow = cMatrix.col(i);
    cMRow.max(max_index);
    predictedLabels(i) = max_index;
  }
}

/**
 * This function helps in building the Weight Distribution matrix which is
 * updated during every iteration. It calculates the "difficulty" in classifying
 * a point by adding the weights for all instances, using D.
 *
 * @param D The 2 Dimensional weight matrix from which the weights are
 *      to be calculated.
 * @param weights The output weight vector.
 */
template <typename MatType, typename WeakLearner>
void AdaBoost<MatType, WeakLearner>::BuildWeightMatrix(
    const arma::mat& D,
    arma::rowvec& weights)
{
  size_t i, j;
  weights.fill(0.0);

  for (i = 0; i < D.n_rows; i++)
  {
    for (j = 0; j < D.n_cols; j++)
      weights(i) += D(i, j);
  }
}

} // namespace adaboost
} // namespace mlpack

#endif
