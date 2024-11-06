/**
 * @file methods/hoeffding_trees/hoeffding_categorical_split.hpp
 * @author Ryan Curtin
 *
 * A class that contains the information necessary to perform a categorical
 * split for Hoeffding trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP
#define MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP

#include <mlpack/prereqs.hpp>
#include "categorical_split_info.hpp"

namespace mlpack {

/**
 * This is the standard Hoeffding-bound categorical feature proposed in the
 * paper below:
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
 * This class will track the sufficient statistics of the training points it has
 * seen.  The HoeffdingSplit class (and other related classes) can use this
 * class to track categorical features and split decision tree nodes.
 *
 * @tparam FitnessFunction Fitness function to use for calculating gain.
 */
template<typename FitnessFunction>
class HoeffdingCategoricalSplit
{
 public:
  //! The type of split information required by the HoeffdingCategoricalSplit.
  using SplitInfo = CategoricalSplitInfo;

  /**
   * Create the HoeffdingCategoricalSplit given a number of categories for this
   * dimension and a number of classes.
   *
   * @param numCategories Number of categories in this dimension.
   * @param numClasses Number of classes in this dimension.
   */
  HoeffdingCategoricalSplit(const size_t numCategories = 0,
                            const size_t numClasses = 0);

  /**
   * Create the HoeffdingCategoricalSplit given a number of categories for this
   * dimension and a number of classes and another HoeffdingCategoricalSplit to
   * take parameters from.  In this particular case, there are no parameters to
   * take, but this constructor is required by the HoeffdingTree class.
   */
  HoeffdingCategoricalSplit(const size_t numCategories,
                            const size_t numClasses,
                            const HoeffdingCategoricalSplit& other);

  /**
   * Train on the given value with the given label.
   *
   * @param value Value to train on.
   * @param label Label to train on.
   */
  template<typename eT>
  void Train(eT value, const size_t label);

  /**
   * Given the points seen so far, evaluate the fitness function, returning the
   * gain for the best possible split and the second best possible split.  In
   * this splitting technique, we only split one possible way, so
   * secondBestFitness will always be 0.
   *
   * @param bestFitness The fitness function result for this split.
   * @param secondBestFitness This is always set to 0 (this split only splits
   *      one way).
   */
  void EvaluateFitnessFunction(double& bestFitness, double& secondBestFitness)
      const;

  //! Return the number of children, if the node were to split.
  size_t NumChildren() const { return sufficientStatistics.n_cols; }

  /**
   * Gather the information for a split: get the labels of the child majorities,
   * and initialize the SplitInfo object.
   *
   * @param childMajorities Majorities of child nodes to be created.
   * @param splitInfo Information for splitting.
   */
  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);

  //! Get the majority class seen so far.
  size_t MajorityClass() const;
  //! Get the probability of the majority class given the points seen so far.
  double MajorityProbability() const;

  //! Serialize the categorical split.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(sufficientStatistics));
  }

 private:
  //! The sufficient statistics for all points seen so far.  Each column
  //! corresponds to a category, and contains a count of each of the classes
  //! seen for points in that category.
  arma::Mat<size_t> sufficientStatistics;
};

} // namespace mlpack

// Include implementation.
#include "hoeffding_categorical_split_impl.hpp"

#endif
