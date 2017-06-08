/**
 * @file bootstrap.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Bootstrap() function, which creates a bootstrapped
 * dataset from the given input dataset.
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
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    const size_t index = math::RandInt(dataset.n_cols);
    bootstrapDataset.col(i) = dataset.col(index);
    bootstrapLabels[i] = labels[index];
    if (UseWeights)
      bootstrapWeights[i] = weights[index];
  }
}

} // namespace tree
} // namespace mlpack

#endif
