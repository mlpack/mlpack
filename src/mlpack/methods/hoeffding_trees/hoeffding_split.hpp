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
         template<typename> class NumericSplitType =
             HoeffdingDoubleNumericSplit,
         template<typename> class CategoricalSplitType =
             HoeffdingCategoricalSplit
>
class HoeffdingSplit
{
 public:
  HoeffdingSplit(const size_t dimensionality,
                 const size_t numClasses,
                 const data::DatasetInfo& datasetInfo,
                 const double successProbability,
                 const size_t maxSamples,
                 std::unordered_map<size_t, std::pair<size_t, size_t>>*
                     dimensionMappings = NULL);

  ~HoeffdingSplit();

  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  // 0 if split should not happen; number of splits otherwise.
  size_t SplitCheck();

  //! Get the splitting dimension (size_t(-1) if no split).
  size_t SplitDimension() const { return splitDimension; }

  //! Get the majority class.
  size_t MajorityClass() const;
  //! Modify the majority class.
  size_t& MajorityClass();

  // Return index that we should go towards.
  template<typename VecType>
  size_t CalculateDirection(const VecType& point) const;

  // Classify the point according to the statistics in this node.
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  template<typename VecType>
  void Classify(const VecType& point, size_t& prediction, double& probability)
      const;

  template<typename StreamingDecisionTreeType>
  void CreateChildren(std::vector<StreamingDecisionTreeType>& children);

  //! Serialize the split.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  // We need to keep some information for before we have split.
  std::vector<NumericSplitType<FitnessFunction>> numericSplits;
  std::vector<CategoricalSplitType<FitnessFunction>> categoricalSplits;

  // This structure is owned by this node only if it is the root of the tree.
  std::unordered_map<size_t, std::pair<size_t, size_t>>* dimensionMappings;
  // Indicates whether or not we own the mappings.
  bool ownsMappings;

  size_t numSamples;
  size_t numClasses;
  size_t maxSamples;
  data::DatasetInfo* datasetInfo;
  double successProbability;

  // And we need to keep some information for after we have split.
  size_t splitDimension;
  size_t majorityClass;
  double majorityProbability;
  // In case it's categorical.
  typename CategoricalSplitType<FitnessFunction>::SplitInfo categoricalSplit;
  // In case it's numeric.
  typename NumericSplitType<FitnessFunction>::SplitInfo numericSplit;
};

} // namespace tree
} // namespace mlpack

#include "hoeffding_split_impl.hpp"

#endif
