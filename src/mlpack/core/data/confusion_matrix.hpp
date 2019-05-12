/**
 * @file confusion_matrix_impl.hpp
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
#ifndef MLPACK_CORE_DATA_CONFUSION_MATRIX_HPP
#define MLPACK_CORE_DATA_CONFUSION_MATRIX_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * A confusion matrix is a summary of prediction results on a classification
 * problem.The number of correct and incorrect predictions are summarized
 * with count values and broken down by each class.
 * for example for 2 classes the function will be
 * confusionmatrix(predictors, responses, output, 2)
 * output matrix will be of size 2 * 2
 *
 *         0     1
 *    0    TP    FN
 *    1    FP    TN
 *
 * Confusion matrix for two labels will look like above.
 * The row contains the predicted values and column contains the actual values.
 *
 * @param predictors Vector of data points.
 * @param responses The measured data for each point.
 * @param output Matrix which is represented as confusion matrix.
 * @param countlables Number of classes.
 */
template<typename eT>
void ConfusionMatrix(const arma::Row<size_t> predictors,
                     const arma::Row<size_t> responses,
                     arma::Mat<eT>& output,
                     const size_t countlabels);
} // namespace data
} // namespace mlpack

// Include implementation.
#include "confusion_matrix_impl.hpp"

#endif
