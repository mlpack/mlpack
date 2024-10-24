/**
 * @file methods/hoeffding_trees/hoeffding_numeric_split.hpp
 * @author Ryan Curtin
 *
 * A numeric feature split for Hoeffding trees.  This is a very simple
 * implementation based on a minor note in the paper "Mining High-Speed Data
 * Streams" by Pedro Domingos and Geoff Hulten.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_NUMERIC_SPLIT_HPP

#include <mlpack/prereqs.hpp>
#include "numeric_split_info.hpp"

namespace mlpack {

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
 *
 * @tparam FitnessFunction Fitness function to use for calculating gain.
 * @tparam ObservationType Type of observations in this dimension.
 */
template<typename FitnessFunction,
         typename ObservationType = double>
class HoeffdingNumericSplit
{
 public:
  //! The splitting information type required by the HoeffdingNumericSplit.
  using SplitInfo = NumericSplitInfo<ObservationType>;

  /**
   * Create the HoeffdingNumericSplit class, and specify some basic parameters
   * about how the binning should take place.
   *
   * @param numClasses Number of classes.
   * @param bins Number of bins.
   * @param observationsBeforeBinning Number of points to see before binning is
   *      performed.
   */
  HoeffdingNumericSplit(const size_t numClasses = 0,
                        const size_t bins = 10,
                        const size_t observationsBeforeBinning = 100);

  /**
   * Create the HoeffdingNumericSplit class, using the parameters from the given
   * other split object.
   */
  HoeffdingNumericSplit(const size_t numClasses,
                        const HoeffdingNumericSplit& other);

  /**
   * Train the HoeffdingNumericSplit on the given observed value (remember that
   * this object only cares about the information for a single feature, not an
   * entire point).
   *
   * @param value Value in the dimension that this HoeffdingNumericSplit refers
   *      to.
   * @param label Label of the given point.
   */
  void Train(ObservationType value, const size_t label);

  /**
   * Evaluate the fitness function given what has been calculated so far.  In
   * this case, if binning has not yet been performed, 0 will be returned (i.e.,
   * no gain).  Because this split can only split one possible way,
   * secondBestFitness (the fitness function for the second best possible split)
   * will be set to 0.
   *
   * @param bestFitness Value of the fitness function for the best possible
   *      split.
   * @param secondBestFitness Value of the fitness function for the second best
   *      possible split (always 0 for this split).
   */
  void EvaluateFitnessFunction(double& bestFitness, double& secondBestFitness)
      const;

  //! Return the number of children if this node splits on this feature.
  size_t NumChildren() const { return bins; }

  /**
   * Return the majority class of each child to be created, if a split on this
   * dimension was performed.  Also create the split object.
   */
  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo) const;

  //! Return the majority class.
  size_t MajorityClass() const;
  //! Return the probability of the majority class.
  double MajorityProbability() const;

  //! Return the number of bins.
  size_t Bins() const { return bins; }

  //! Serialize the object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Before binning, this holds the points we have seen so far.
  arma::Col<ObservationType> observations;
  //! This holds the labels of the points before binning.
  arma::Col<size_t> labels;

  //! The split points for the binning (length bins - 1).
  arma::Col<ObservationType> splitPoints;
  //! The number of bins.
  size_t bins;
  //! The number of observations we must see before binning.
  size_t observationsBeforeBinning;
  //! The number of samples we have seen so far.
  size_t samplesSeen;

  //! After binning, this contains the sufficient statistics.
  arma::Mat<size_t> sufficientStatistics;
};

//! Convenience typedef.
template<typename FitnessFunction>
using HoeffdingDoubleNumericSplit = HoeffdingNumericSplit<FitnessFunction,
    double>;

template<typename FitnessFunction>
using HoeffdingFloatNumericSplit = HoeffdingNumericSplit<FitnessFunction,
    float>;

} // namespace mlpack

// Include implementation.
#include "hoeffding_numeric_split_impl.hpp"

#endif
