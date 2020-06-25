/**
 * @file core/data/correlation_feature_selection.hpp
 * @author Jeffin Sam
 *
 * Feature selction based on T-Test.
 * The methodology used in the proposed algorithm is to use t-test
 * for selecting the most significant features from the features set.
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CORRELATION_FEATURE_SELECTION_IMPL_HPP
#define MLPACK_CORE_DATA_CORRELATION_FEATURE_SELECTION_IMPL_HPP

// In case it hasn't been included yet.
#include "correlation_feature_selection.hpp"

namespace mlpack {
namespace data {
namespace fs {

/**
 *
 * @param input Input dataset with actual number of features.
 * @param target Ouput labels for the respective Input. 
 * @param output Output matrix with lesser number of features.
 * @param outputSize No of features you want in output matrix.
 */
template<typename T>
void CorrelationSelection(const arma::Mat<T>& input,
                          const arma::rowvec target,
                          arma::Mat<T>& output,
                          const size_t outputSize)
{
  std::vector<std::vector<double>> outputIndex(input.n_rows,
      std::vector<double> (2));

  for (size_t i = 0; i < input.n_rows; i++)
  {
    double rValue = arma::mat(arma::cor(input.row(i), target)).at(0);
    if (std::isfinite(rValue))
    {
      double tValue = rValue * (pow((input.n_cols - 2) / (1 - pow(rValue,
          2)), 0.5));
      outputIndex[i][0] = tValue;
      outputIndex[i][1] = i;
    }
    else
    {
      outputIndex[i][0] = std::numeric_limits<double>::lowest();
      outputIndex[i][1] = i;
    }
  }
  sort(outputIndex.rbegin(), outputIndex.rend());
  std::vector<long long unsigned int> indices;

  for (size_t i = 0; i < std::min(outputSize, outputIndex.size()); i++)
    indices.push_back((long long unsigned int)outputIndex[i][1]);

  output = input.rows(arma::uvec(indices));
}

} // namespace fs
} // namespace data
} // namespace mlpack

#endif
