/**
 * @file all_categorical_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the AllCategoricalSplit categorical split class.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_ALL_CATEGORICAL_SPLIT_IMPL_HPP

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
template<typename VecType>
double AllCategoricalSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const size_t numCategories,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t minimumLeafSize,
    arma::Col<typename VecType::elem_type>& classProbabilities,
    AllCategoricalSplit::AuxiliarySplitInfo<typename VecType::elem_type>& aux)
{
  // Count the number of elements in each potential child.
  arma::Col<size_t> counts(numCategories);
  counts.zeros();
  for (size_t i = 0; i < data.n_elem; ++i)
    counts[(size_t) data[i]]++;

  // If each child will have the minimum number of points in it, we can split.
  // Otherwise we can't.
  if (arma::min(counts) < minimumLeafSize)
    return bestGain;

  // Calculate the gain of the split.  First we have to calculate the labels
  // that would be assigned to each child.
  arma::uvec childPositions(numCategories);
  std::vector<arma::Row<size_t>> childLabels;
  for (size_t i = 0; i < numCategories; ++i)
    childLabels[i].zeros(counts[i]);

  // Extract labels for each child.
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    const size_t category = (size_t) data[i];
    childLabels[category][childPositions[category]++] = labels[i];
  }

  double overallGain = 0.0;
  for (size_t i = 0; i < counts.n_elem; ++i)
  {
    // Calculate the gain of this child.
    const double childPct = double(counts[i]) / double(data.n_elem);
    const double childGain = FitnessFunction::Evaluate(childLabels[i],
        numClasses);

    overallGain += childPct * childGain;
  }

  if (overallGain > bestGain)
  {
    // This is better, so set up the class probabilities vector and return.
    classProbabilities.set_size(1);
    classProbabilities[0] = numCategories;
    return overallGain;
  }

  // Otherwise there was no improvement.
  return bestGain;
}

template<typename FitnessFunction>
template<typename ElemType>
size_t AllCategoricalSplit<FitnessFunction>::NumChildren(
    const arma::Col<ElemType>& classProbabilities,
    const AllCategoricalSplit::AuxiliarySplitInfo<ElemType>& /* aux */)
{
  return classProbabilities[0];
}

template<typename FitnessFunction>
template<typename ElemType>
size_t AllCategoricalSplit<FitnessFunction>::CalculateDirection(
    const ElemType& point,
    const arma::Col<ElemType>& classProbabilities,
    const AllCategoricalSplit::AuxiliarySplitInfo<ElemType>& /* aux */)
{
  return point;
}

} // namespace tree
} // namespace mlpack

#endif
