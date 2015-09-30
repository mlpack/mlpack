/**
 * @file hoeffding_numeric_split.hpp
 * @author Ryan Curtin
 *
 * A numeric feature split for Hoeffding trees.  This is a very simple
 * implementation based on a minor note in the paper "Mining High-Speed Data
 * Streams" by Pedro Domingos and Geoff Hulten.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP

#include <mlpack/core.hpp>
#include "numeric_split_info.hpp"

namespace mlpack {
namespace tree {

/**
 * The HoeffdingNumericSplit class implements the numeric feature splitting
 * strategy alluded to by Domingos and Hulten in the following paper:
 *
 * @code
 * @inproceedings{domingos2000mining,
 *   title={{Mining High-Speed Data Streams}},
 *   author={Domingos, P. and Hulten, G.},
 *   year={2000},
 *   booktitle={Proceedings of the Sixth ACM SIGKDD International Conference on
 *       Knowledge Discovery and Data Mining (KDD '00)},
 *   pages={71--80}
 * }
 * @endcode
 *
 * The strategy alluded to is very simple: we discretize the numeric features
 * that we see.  But in this case, we don't know how many bins we have, which
 * makes things a little difficult.  This class only makes binary splits, and
 * has a maximum number of bins.  The binning strategy is simple: the split
 * caches the minimum and maximum value of points seen so far, and when the
 * number of points hits a predefined threshold, the cached minimum-maximum
 * range is equally split into bins, and splitting proceeds in the same way as
 * with the categorical splits.  This is a simple and stupid strategy, so don't
 * expect it to be the best possible thing you can do.
 */
template<typename FitnessFunction,
         typename ObservationType = double>
class HoeffdingNumericSplit
{
 public:
  typedef NumericSplitInfo<ObservationType> SplitInfo;

  HoeffdingNumericSplit(const size_t numClasses,
                        const size_t bins = 10,
                        const size_t observationsBeforeBinning = 100);

  void Train(ObservationType value, const size_t label);

  double EvaluateFitnessFunction() const;

  // Return the majority class of each child to be created, if a split on this
  // dimension was performed.  Also create the split object.
  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo) const;

  size_t MajorityClass() const;

  size_t Bins() const { return bins; }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  // Cache the values of the points seen before we make bins.
  arma::Col<ObservationType> observations;
  arma::Col<size_t> labels;

  arma::Col<ObservationType> splitPoints;
  size_t bins;
  size_t observationsBeforeBinning;
  size_t samplesSeen;

  arma::Mat<size_t> sufficientStatistics;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "hoeffding_numeric_split_impl.hpp"

#endif
