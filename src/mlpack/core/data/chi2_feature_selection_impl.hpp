/**
 * @file core/data/chi2_feature_selection_impl.hpp
 * @author Jeffin Sam
 *
 * Feature selction based on Chi-Square Test.
 * This test thus can be used to determine the best features for a given
 * dataset by determining the features on which the output class label
 * is most dependent on.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CHI2_FEATURE_SELECTION_IMPL_HPP
#define MLPACK_CORE_DATA_CHI2_FEATURE_SELECTION_IMPL_HPP

// In case it hasn't been included yet.
#include "chi2_feature_selection.hpp"

namespace mlpack {
namespace data {
namespace fs {

/**
 * The function takes an input dataset with a set number of features,
 * and the required number of features along with the target varibale 
 * and output the dataset with lesser number of features.
 * 
 * @param input Input dataset with actual number of features.
 * @param target Output labels for the respective Input. 
 * @param output Output matrix with lesser number of features.
 * @param outputSize Number of features in the output matrix.
 */
template<typename T>
void Chi2Selection(const arma::Mat<T>& input,
                   const arma::rowvec target,
                   arma::Mat<T>& output,
                   const size_t outputSize)
{
  // Calculate unique target labels.
  arma::rowvec outputLabels = arma::unique(target);
  if (outputSize > input.n_rows)
  {
    throw std::runtime_error("Output size should be less than "
        "input size!");
  }
  if (target.n_cols != input.n_cols)
  {
    throw std::runtime_error("Number of columns in target and input "
        "does not match, Please verify!");
  }
  std::vector<std::vector<double>> outputIndex(input.n_rows,
      std::vector<double> (2));
  for (size_t i = 0; i < input.n_rows; i++)
  {
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> chiTable;
    std::unordered_map<size_t, size_t>labels, expected;
    for (size_t j = 0; j < input.n_cols; j++)
    {
     chiTable[input.at(i, j)][target(j)]++;
     labels[target(j)]++;
     expected[input.at(i, j)]++;
    }

    // For safety, lets add a padding.
    for (auto it = chiTable.begin(); it != chiTable.end(); it++)
    {
      for (size_t i = 0; i < outputLabels.n_rows; i++)
      {
        if (it->second.find(outputLabels(i)) == it->second.end())
          chiTable[it->first][outputLabels(i)] = 0;
      }
    }

    // Calculate chi square values.
    double chiValue = 0.0;
    for (auto it = chiTable.begin(); it != chiTable.end(); it++)
    {
      for (auto jt = it->second.begin(); jt != it->second.end(); jt++)
      {
        chiValue +=
            (pow((jt->second - ((double)(expected[it->first] *
            labels[jt->first]) / (double)(input.n_cols))), 2) /
            ((double)(expected[it->first] * labels[jt->first]) /
            (double)(input.n_cols)));
      }
    }

    outputIndex[i][0] = chiValue;
    outputIndex[i][1] = i;
  }
  sort(outputIndex.rbegin(), outputIndex.rend());
  std::vector<unsigned long long> indices;

  for (size_t i = 0; i < std::min(outputSize, outputIndex.size()); i++)
    indices.push_back((unsigned long long)outputIndex[i][1]);

  output = input.rows(arma::uvec(indices));
}

} // namespace fs
} // namespace data
} // namespace mlpack

#endif
