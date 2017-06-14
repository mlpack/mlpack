/**
 * @file accuracy_impl.hpp
 * @author Kirill Mishchenko
 *
 * The implementation of the class Accuracy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ACCURACY_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_ACCURACY_IMPL_HPP

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType>
double Accuracy::Evaluate(MLAlgorithm& model,
                          const DataType& data,
                          const arma::Row<size_t>& labels)
{
  if (data.n_cols != labels.n_elem)
  {
    std::ostringstream oss;
    oss << "Accuracy::Evaluate(): number of points (" << data.n_cols << ") "
        << "does not match number of labels (" << labels.n_elem << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);
  size_t amountOfCorrectPredictions = arma::sum(predictedLabels == labels);

  return (double) amountOfCorrectPredictions / labels.n_elem;
}

} // namespace cv
} // namespace mlpack

#endif
