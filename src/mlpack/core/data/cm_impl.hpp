/**
 * @file cm_impl.hpp
 * @author Jeffin Sam
 *
 * implementation of confusion matrix
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CM_IMPL_HPP
#define MLPACK_CORE_DATA_CM_IMPL_HPP

// In case it hasn't been included yet.
#include "cm.hpp"

namespace mlpack {
namespace data {

/**
 * A confusion matrix is a summary of prediction results on a classification
 * problem.The number of correct and incorrect predictions are summarized
 * with count values and broken down by each class.
 *
 * @param pred vector of predicted values.
 * @param actual vector of actual values.
 * @param output matrix which is represnted as confusion matrix.
 * @param countlables no of classes
 * for example for 2 classes the function will be
 * confusionmatrix(pred,actual,matrix,2)
 * output matrix will be of size 2 * 2
 *         0     1
 *    0    TP    FN
 *    1    FP    TN
 * confusion matrix for two labels will look like above.
 * Row is the predicted values and column are actual values.
 */
template<typename eT, typename RowType>
void ConfusionMatrix(const RowType& pred,
                     const RowType& actual,
                     arma::Mat<eT>& output,
                     const size_t countlabels)
{
  // finding whether continues or not
  bool find = true;
  for (size_t i = 0; i < pred.n_elem; ++i)
  {
    if (pred[i] != int(pred[i]) || actual[i] != int(actual[i]))
    {
      // simply means continues data or different datatype
      find = false;
      break;
    }
  }
  if (find == false)
  {
    throw std::runtime_error("Datatset should be discrete");
  }
  // Loop over the input labels and predicted and just add the count
  output.set_size(countlabels, countlabels);
  output.fill(0);
  for (size_t i = 0; i < pred.n_elem; ++i)
  {
    output(pred[i], actual[i]) = output(pred[i], actual[i]) + 1;
  }
}
} // namespace data
} // namespace mlpack

#endif
