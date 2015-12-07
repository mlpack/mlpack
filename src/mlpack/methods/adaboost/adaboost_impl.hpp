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
 * Constructor. Currently runs the AdaBoost.MH algorithm.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param iterations Number of boosting rounds
 * @param tol Tolerance for termination of Adaboost.MH.
 * @param other Weak Learner, which has been initialized already.
 */
template<typename MatType, typename WeakLearner>
AdaBoost<MatType, WeakLearner>::AdaBoost(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const WeakLearner& other,
    const size_t iterations,
    const double tol)
{
  Train(data, labels, other, iterations, tol);
}

// Train AdaBoost.
template<typename MatType, typename WeakLearner>
void AdaBoost<MatType, WeakLearner>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const WeakLearner& other,
    const size_t iterations,
    const double tolerance)
{
  // Clear information from previous runs.
  wl.clear();
  alpha.clear();

  // Count the number of classes.
  classes = (arma::max(labels) - arma::min(labels)) + 1;
  this->tolerance = tolerance;

  // crt is the cumulative rt value for terminating the optimization when rt is
  // changing by less than the tolerance.
  double rt, crt, alphat = 0.0, zt;

  ztProduct = 1.0;

  // To be used for prediction by the weak learner.
  arma::Row<size_t> predictedLabels(labels.n_cols);

  // Use tempData to modify input data for incorporating weights.
  MatType tempData(data);

  // This matrix is a helper matrix used to calculate the final hypothesis.
  arma::mat sumFinalH = arma::zeros<arma::mat>(classes, predictedLabels.n_cols);

  // Load the initial weights into a 2-D matrix.
  const double initWeight = 1.0 / double(data.n_cols * classes);
  arma::mat D(classes, data.n_cols);
  D.fill(initWeight);

  // Weights are stored in this row vector.
  arma::rowvec weights(predictedLabels.n_cols);

  // This is the final hypothesis.
  arma::Row<size_t> finalH(predictedLabels.n_cols);

  // Now, start the boosting rounds.
  for (size_t i = 0; i < iterations; i++)
  {
    // Initialized to zero in every round.  rt is used for calculation of
    // alphat; it is the weighted error.
    // rt = (sum) D(i) y(i) ht(xi)
    rt = 0.0;

    // zt is used for weight normalization.
    zt = 0.0;

    // Build the weight vectors.
    weights = arma::sum(D);

    // Use the existing weak learner to train a new one with new weights.
    WeakLearner w(other, tempData, labels, weights);
    w.Classify(tempData, predictedLabels);

    // Now from predictedLabels, build ht, the weak hypothesis
    // buildClassificationMatrix(ht, predictedLabels);

    // Now, calculate alpha(t) using ht.
    for (size_t j = 0; j < D.n_cols; j++) // instead of D, ht
    {
      if (predictedLabels(j) == labels(j))
        rt += arma::accu(D.col(j));
      else
        rt -= arma::accu(D.col(j));
    }

    if ((i > 0) && (std::abs(rt - crt) < tolerance))
      break;

    crt = rt;

    // Our goal is to find alphat which mizimizes or approximately minimizes the
    // value of Z as a function of alpha.
    alphat = 0.5 * log((1 + rt) / (1 - rt));

    alpha.push_back(alphat);
    wl.push_back(w);

    // Now start modifying the weights.
    for (size_t j = 0; j < D.n_cols; j++)
    {
      const double expo = exp(alphat);
      if (predictedLabels(j) == labels(j))
      {
        for (size_t k = 0; k < D.n_rows; k++)
        {
          // We calculate zt, the normalization constant.
          D(k, j) /= expo;
          zt += D(k, j); // * exp(-1 * alphat * yt(j,k) * ht(j,k));


          // Add to the final hypothesis matrix.
          // sumFinalH(k, j) += (alphat * ht(k, j));
          if (k == labels(j))
            sumFinalH(k, j) += (alphat); // * ht(k, j));
          else
            sumFinalH(k, j) -= (alphat);
        }
      }
      else
      {
        for (size_t k = 0; k < D.n_rows; k++)
        {
          // We calculate zt, the normalization constant.
          D(k, j) *= expo;
          zt += D(k, j);

          // Add to the final hypothesis matrix.
          if (k == labels(j))
            sumFinalH(k, j) += alphat; // * ht(k, j));
          else
            sumFinalH(k, j) -= alphat;
        }
      }
    }

    // Normalize D.
    D /= zt;

    // Accumulate the value of zt for the Hamming loss bound.
    ztProduct *= zt;
  }

  // Iterations are over, now build a strong hypothesis from a weighted
  // combination of these weak hypotheses.
  arma::colvec tempSumFinalH;
  arma::uword maxIndex;

  for (size_t i = 0;i < sumFinalH.n_cols; i++)
  {
    tempSumFinalH = sumFinalH.unsafe_col(i);
    tempSumFinalH.max(maxIndex);
    finalH(i) = maxIndex;
  }

  finalHypothesis = finalH;
}

/**
 * Classify the given test points.
 */
template <typename MatType, typename WeakLearner>
void AdaBoost<MatType, WeakLearner>::Classify(
    const MatType& test,
    arma::Row<size_t>& predictedLabels)
{
  arma::Row<size_t> tempPredictedLabels(test.n_cols);
  arma::mat cMatrix(classes, test.n_cols);

  cMatrix.zeros();
  predictedLabels.set_size(test.n_cols);

  for (size_t i = 0; i < wl.size(); i++)
  {
    wl[i].Classify(test, tempPredictedLabels);

    for (size_t j = 0; j < tempPredictedLabels.n_cols; j++)
      cMatrix(tempPredictedLabels(j), j) += alpha[i];
  }

  arma::colvec cMRow;
  arma::uword maxIndex;

  for (size_t i = 0; i < predictedLabels.n_cols; i++)
  {
    cMRow = cMatrix.unsafe_col(i);
    cMRow.max(maxIndex);
    predictedLabels(i) = maxIndex;
  }
}

} // namespace adaboost
} // namespace mlpack

#endif
