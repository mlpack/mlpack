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
                        const TopK& k)
{
  if (data.n_cols != labels.n_cols)
  {
    std::ostringstream fpp;
    fpp << "topK_accuracy_score::Evaluate(): number of inputs ("<< data.n_cols <<")"
        << "does not matches with number of target labels ("<< labels.n_cols <<")" 
        << std::endl;
    throw std::invalid_argument(fpp.str());
  }
  arma::Row<size_t> predictedLabels;
  // Taking Classification output from the model.
  model.Classify(data, predictedLabels);
  // Shape of the predicted values.
  double predictedClass = predictedLabels.size();
  // Top 'k' label prediction class. 
  if (k < predictedClass[1])
  {
    const int idx = predictedClass[1] - k - 1;
  else
  {
    std::ostringstream fs;
    fs << "There are less number of class for Top K Accuracy score" << std::endl;
  }
  size_t idx = predictedClass[1] - k - 1;
  float count = 0;
  double srt = sort_index(predictedLabels);  
  for (i = 1; i < predictedClass[0]; i++)
  {
    if (labels[i] != srt[i, idx+1:])
    {
      count = count + 1;
    }
  }
  // Accuracy Score of top k predicted class labels.
  return (double) count / predictedClass[0];
}
} // namespace cv
} // namespace mlpack
#endif

