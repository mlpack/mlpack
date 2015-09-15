/**
 * @file hoeffding_split.hpp
 * @author Ryan Curtin
 *
 * An implementation of the standard Hoeffding bound split by Pedro Domingos and
 * Geoff Hulten in ``Mining High-Speed Data Streams''.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_SPLIT_HPP

namespace mlpack {
namespace tree {

template<typename FitnessFunction,
         typename NumericSplitType,
         typename CategoricalSplitType>
class HoeffdingSplit
{
 public:
  HoeffdingSplit(const size_t dimensionality,
                 const size_t numClasses,
                 const DatasetInfo& datasetInfo);

  template<typename VecType>
  void Train(VecType& point, const size_t label);

  // 0 if split should not happen; number of splits otherwise.
  size_t SplitCheck() const;

  // Return index that we should go towards.
  template<typename VecType>
  size_t CalculateDirection(VecType& point) const;

 private:
  // We need to keep some information for before we have split.
  std::vector<NumericSplitType> numericSplits;
  std::vector<CategoricalSplitType> categoricalSplits;

  const DatasetInfo& datasetInfo;

  // And we need to keep some information for after we have split.
  size_t splitDimension;
  typename CategoricalSplitType::SplitInfo categoricalSplit; // In case it's categorical.
  typename NumericSplitType::SplitInfo numericSplit; // In case it's numeric.
};

} // namespace tree
} // namespace mlpack

#endif
