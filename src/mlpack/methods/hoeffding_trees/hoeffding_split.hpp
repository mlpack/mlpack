/**
 * @file hoeffding_split.hpp
 * @author Ryan Curtin
 *
 * An implementation of the standard Hoeffding bound split by Pedro Domingos and
 * Geoff Hulten in ``Mining High-Speed Data Streams''.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_SPLIT_HPP

#include <mlpack/core.hpp>
#include "gini_impurity.hpp"
#include "hoeffding_numeric_split.hpp"
#include "hoeffding_categorical_split.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction = GiniImpurity,
         typename NumericSplitType = HoeffdingNumericSplit<GiniImpurity>,
         typename CategoricalSplitType = HoeffdingCategoricalSplit<GiniImpurity>
>
class HoeffdingSplit
{
 public:
  HoeffdingSplit(const size_t dimensionality,
                 const size_t numClasses,
                 const data::DatasetInfo& datasetInfo,
                 const double successProbability);

  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  // 0 if split should not happen; number of splits otherwise.
  size_t SplitCheck();

  // Return index that we should go towards.
  template<typename VecType>
  size_t CalculateDirection(const VecType& point) const;

  // Classify the point according to the statistics in this node.
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  template<typename StreamingDecisionTreeType>
  void CreateChildren(std::vector<StreamingDecisionTreeType>& children);

 private:
  // We need to keep some information for before we have split.
  std::vector<NumericSplitType> numericSplits;
  std::vector<CategoricalSplitType> categoricalSplits;

  size_t numSamples;
  size_t numClasses;
  const data::DatasetInfo& datasetInfo;
  double successProbability;

  // And we need to keep some information for after we have split.
  size_t splitDimension;
  size_t majorityClass;
  typename CategoricalSplitType::SplitInfo categoricalSplit; // In case it's categorical.
  typename NumericSplitType::SplitInfo numericSplit; // In case it's numeric.
};

} // namespace tree
} // namespace mlpack

#include "hoeffding_split_impl.hpp"

#endif
