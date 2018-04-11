/**
 * @file bootstrap.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Bootstrap() function, which creates a bootstrapped
 * dataset from the given input dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_BOOTSTRAP_HPP
#define MLPACK_METHODS_RANDOM_FOREST_BOOTSTRAP_HPP

namespace mlpack {
namespace tree {

/**
 * Given a dataset, create another dataset via bootstrap sampling, with labels.
 */
template<bool UseWeights,
         typename MatType,
         typename LabelsType,
         typename WeightsType>
void Bootstrap(const MatType& dataset,
               const LabelsType& labels,
               const WeightsType& weights,
               MatType& bootstrapDataset,
               LabelsType& bootstrapLabels,
               WeightsType& bootstrapWeights)
{
  bootstrapDataset.set_size(dataset.n_rows, dataset.n_cols);
  bootstrapLabels.set_size(labels.n_elem);
  if (UseWeights)
    bootstrapWeights.set_size(weights.n_elem);

  // Random sampling with replacement.
  arma::uvec indices = arma::randi<arma::uvec>(dataset.n_cols,
      arma::distr_param(0, dataset.n_cols - 1));
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    bootstrapDataset.col(i) = dataset.col(indices[i]);
    bootstrapLabels[i] = labels[indices[i]];
    if (UseWeights)
      bootstrapWeights[i] = weights[indices[i]];
  }
}

} // namespace tree
} // namespace mlpack

#endif
