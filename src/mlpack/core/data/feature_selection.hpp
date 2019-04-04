/**
 * @file feature_selection_imp.hpp
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
#ifndef MLPACK_CORE_DATA_FEATURE_SELECTION_HPP
#define MLPACK_CORE_DATA_FEATURE_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

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
void SelectBestFeature(const arma::Mat<T>& input,
              const double threshold,
              arma::Mat<T>& output);

} // namespace data
} // namespace mlpack

// Include implementation.
#include "feature_selection_imp.hpp"

#endif
