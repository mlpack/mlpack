/**
 * @file top_k_accuracy_impl.hpp
 * @author Arunav Shandeelya
 * 
 * The implementation of class Top_K_Accuracy_score
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_IMPL_HPP

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType, typename TopK>
double TopKAccuracy::Evaluate(MLAlgorithm& model,
                        const DataType& data,
                        const arma::Row<size_t> labels,
                        const DataType& predsprob,
                        const TopK& k)
{
  if (data.n_cols != labels.n_cols)
  {
    std::ostringstream fpp;
    fpp << "TopK_Accuracy::Evaluate(): number of points (" << data.n_cols << ")"
        << "doesn't matches number of labels (" << labels.n_cols << ")!"
        << std::endl;
    throw std::invalid_argument(fpp.str());
  }
  // Taking Classification output from the model.
  model.Classify(data, predsprob);
  // Shape of the predicted probabilities.
  double predictedProb = predsprob.size();
  // Top 'k' label prediction probabilities.
  if (k < predsprob.n_cols)
  {
    size_t idx = predictedProb.n_cols - k - 1;
  else
  {
    std::ostringstream fs;
    fs << "Less number of class for Top K Accuracy score" << std::endl;
  }
  float count = 0;
  size_t srt = arma::sort_index(predsprob);
  for (size_t i = 1; i < predictedProb.n_elem; i++)
  {
    if (labels[i] != arma::span(i, idx + 1), srma::span:;all)
    {
      count = count + 1;
    }
  }
  // Accuracy Score of top k predicted class labels.
  return (double) count / predictedProb.n_elem;
}
} // namespace cv
} // namespace mlpack

#endif
