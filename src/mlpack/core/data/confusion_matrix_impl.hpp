/**
 * @file core/data/confusion_matrix_impl.hpp
 * @author Jeffin Sam
 *
 * Compute confusion matrix to evaluate the accuracy of a classification.
 * The function works only for discrete data/categorical data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CONFUSION_MATRIX_IMPL_HPP
#define MLPACK_CORE_DATA_CONFUSION_MATRIX_IMPL_HPP

// In case it hasn't been included yet.
#include "confusion_matrix.hpp"

namespace mlpack {
namespace data {

/**
 * A confusion matrix is a summary of prediction results on a classification
 * problem.  The number of correct and incorrect predictions are summarized
 * by count and broken down by each class.
 * For example, for 2 classes, the function call will be
 *
 * @code
 * ConfusionMatrix(predictors, responses, output, 2)
 * @endcode
 *
 * In this case, the output matrix will be of size 2 * 2:
 *
 * @code
 *         0     1
 *    0    TP    FN
 *    1    FP    TN
 * @endcode
 *
 * The confusion matrix for two labels will look like what is shown above.  In
 * this confusion matrix, TP represents the number of true positives, FP
 * represents the number of false positives, FN represents the number of false
 * negatives, and TN represents the number of true negatives.
 *
 * When generalizing to 2 or more classes, the row index of the confusion matrix
 * represents the predicted classes and column index represents the actual
 * class.
 */
template<typename eT>
void ConfusionMatrix(const arma::Row<size_t> predictedClasses,
                     const arma::Row<size_t> actualClasses,
                     arma::Mat<eT>& output,
                     const size_t numClasses)
{
  // Loop over the actual labels and predicted labels and add the count.
  output = arma::zeros<arma::Mat<eT> >(numClasses, numClasses);
  for (size_t i = 0; i < predictedClasses.n_elem; ++i)
  {
    output.at(predictedClasses[i], actualClasses[i])++;
  }
}

/**
 * This method calculates the percentages for each of the matrix cell where row
 * index represents predicted class and column index represents actual class.
 */
template<typename eT>
void ConfusionMatrixPercentage(const arma::Row<size_t> predictedClasses,
                               const arma::Row<size_t> actualClasses,
                               arma::Mat<eT>& output,
                               const size_t numClasses)
{
  // Loop over the actual labels and predicted labels and add the count.
  output = arma::zeros<arma::Mat<eT> >(numClasses, numClasses);
  // Count the total number of data points for each of the actual classes.
  arma::Row<double> counts(numClasses, arma::fill::zeros);
  for (size_t i = 0; i < predictedClasses.n_elem; ++i)
  {
    output.at(predictedClasses[i], actualClasses[i])++;
    counts.at(actualClasses[i])++;
  }
  for (size_t i = 0; i < numClasses; ++i)
  {
    for (size_t j = 0; j < numClasses; ++j)
    {
      if (counts[j] > 0)
        output.at(i, j) /= counts[j];
      output.at(i, j) *= 100.0;
    }
  }
}

} // namespace data
} // namespace mlpack

#endif
