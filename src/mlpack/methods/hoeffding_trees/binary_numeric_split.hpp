/**
 * @file binary_numeric_split.hpp
 * @author Ryan Curtin
 *
 * An implementation of the binary-tree-based numeric splitting procedure
 * described by Gama, Rocha, and Medas in their KDD 2003 paper.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_SPLIT_BINARY_NUMERIC_SPLIT_HPP
#define MLPACK_METHODS_HOEFFDING_SPLIT_BINARY_NUMERIC_SPLIT_HPP

#include "binary_numeric_split_info.hpp"

namespace mlpack {
namespace tree {

/**
 * The BinaryNumericSplit class implements the numeric feature splitting
 * strategy devised by Gama, Rocha, and Medas in the following paper:
 *
 * @code
 * @inproceedings{gama2003accurate,
 *    title={Accurate Decision Trees for Mining High-Speed Data Streams},
 *    author={Gama, J. and Rocha, R. and Medas, P.},
 *    year={2003},
 *    booktitle={Proceedings of the Ninth ACM SIGKDD International Conference on
 *        Knowledge Discovery and Data Mining (KDD '03)},
 *    pages={523--528}
 * }
 * @endcode
 *
 * This splitting procedure builds a binary tree on points it has seen so far,
 * and then EvaluateFitnessFunction() returns the best possible split in O(n)
 * time, where n is the number of samples seen so far.  Every split with this
 * split type returns only two splits (greater than or equal to the split point,
 * and less than the split point).  The Train() function should take O(1) time.
 *
 * @tparam FitnessFunction Fitness function to use for calculating gain.
 * @tparam ObservationType Type of observation used by this dimension.
 */
template<typename FitnessFunction,
         typename ObservationType = double>
class BinaryNumericSplit
{
 public:
  //! The splitting information required by the BinaryNumericSplit.
  typedef BinaryNumericSplitInfo<ObservationType> SplitInfo;

  /**
   * Create the BinaryNumericSplit object with the given number of classes.
   *
   * @param numClasses Number of classes in dataset.
   */
  BinaryNumericSplit(const size_t numClasses);

  /**
   * Create the BinaryNumericSplit object with the given number of classes,
   * using information from the given other split for other parameters.  In this
   * case, there are no other parameters, but this function is required by the
   * HoeffdingTree class.
   */
  BinaryNumericSplit(const size_t numClasses, const BinaryNumericSplit& other);

  /**
   * Train on the given value with the given label.
   *
   * @param value The value to train on.
   * @param label The label to train on.
   */
  void Train(ObservationType value, const size_t label);

  /**
   * Given the points seen so far, evaluate the fitness function, returning the
   * best possible gain of a binary split.  Note that this takes O(n) time,
   * where n is the number of points seen so far.  So this may not exactly be
   * fast...
   *
   * The best possible split will be stored in bestFitness, and the second best
   * possible split will be stored in secondBestFitness.
   *
   * @param bestFitness Fitness function value for best possible split.
   * @param secondBestFitness Fitness function value for second best possible
   *      split.
   */
  void EvaluateFitnessFunction(double& bestFitness,
                               double& secondBestFitness);

  // Return the number of children if this node were to split on this feature.
  size_t NumChildren() const { return 2; }

  /**
   * Given that a split should happen, return the majority classes of the (two)
   * children and an initialized SplitInfo object.
   *
   * @param childMajorities Majority classes of the children after the split.
   * @param splitInfo Split information.
   */
  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);

  //! The majority class of the points seen so far.
  size_t MajorityClass() const;
  //! The probability of the majority class given the points seen so far.
  double MajorityProbability() const;

  //! Serialize the object.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! The elements seen so far, in sorted order.
  std::multimap<ObservationType, size_t> sortedElements;
  //! The classes we have seen so far (for majority calculations).
  arma::Col<size_t> classCounts;

  //! A cached best split point.
  ObservationType bestSplit;
  //! If true, the cached best split point is accurate (that is, we have not
  //! seen any more samples since we calculated it).
  bool isAccurate;
};

// Convenience typedef.
template<typename FitnessFunction>
using BinaryDoubleNumericSplit = BinaryNumericSplit<FitnessFunction, double>;

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "binary_numeric_split_impl.hpp"

#endif
