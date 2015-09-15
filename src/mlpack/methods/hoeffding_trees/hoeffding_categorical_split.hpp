/**
 * @file hoeffding_categorical_split.hpp
 * @author Ryan Curtin
 *
 * A class that contains the information necessary to perform a categorical
 * split for Hoeffding trees.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
class HoeffdingCategoricalSplit
{
 public:
  HoeffdingCategoricalSplit(const size_t numCategories, const size_t numClasses);

  template<typename eT>
  void Train(eT value, const size_t label);

  double EvaluateFitnessFunction() const;
 private:
  arma::Mat<size_t> sufficientStatistics;
};

} // namespace tree
} // namespace mlpack

#endif
