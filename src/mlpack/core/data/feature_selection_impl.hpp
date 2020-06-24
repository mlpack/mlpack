/**
 * @file core/data/feature_selection_impl.hpp
 * @author Jeffin Sam
 *
 * Feature selction based on variance thresholding.
 * Motivated by the idea that low variance features contain less
 * information.
 * Calculate varience of each feature, then drop features 
 * with variance below some threshold
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_FEATURE_SELECTION_IMP_HPP
#define MLPACK_CORE_DATA_FEATURE_SELECTION_IMP_HPP

// In case it hasn't been included yet.
#include "feature_selection.hpp"

namespace mlpack {
namespace data {
namespace fs {

/**
 *
 * Feature selector that removes all low-variance features.
 * The idea is when a feature doesn’t vary much within itself,
 * it generally has very little predictive power.
 * Variance Threshold doesn’t consider the relationship of
 * features with the target variable.
 *
 * @param input Input dataset with actual number of features.
 * @param threshold Threshold for variance.
 * @param output Output matrix with lesser number of features.
 */
template<typename T>
void VarianceSelection(const arma::Mat<T>& input, const double threshold,
                       arma::Mat<T>& output)
{
  // Calculate variance of each scaled feature.
  arma::colvec value = arma::var(arma::normalise(input, 2, 1), 0, 1);
  std::vector<long long unsigned int> indices;
  for (size_t i = 0; i < value.n_elem; i++)
  {
    if (value(i) > threshold)
      indices.push_back(i);
  }
  output = input.rows(arma::uvec(indices));
}

} // namespace fs
} // namespace data
} // namespace mlpack

#endif
