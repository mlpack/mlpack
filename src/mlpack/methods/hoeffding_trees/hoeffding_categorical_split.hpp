/**
 * @file hoeffding_categorical_split.hpp
 * @author Ryan Curtin
 *
 * A class that contains the information necessary to perform a categorical
 * split for Hoeffding trees.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_CATEGORICAL_SPLIT_HPP

#include <mlpack/core.hpp>
#include "categorical_split_info.hpp"

namespace mlpack {
namespace tree {

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
  typedef CategoricalSplitInfo SplitInfo;

  /**
   * Create the HoeffdingCategoricalSplit given a number of categories for this
   * dimension and a number of classes.
   *
   * @param numCategories Number of categories in this dimension.
   * @param numClasses Number of classes in this dimension.
   */
  HoeffdingCategoricalSplit(const size_t numCategories,
                            const size_t numClasses);

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
   * gain if a split was to be made.
   */
  double EvaluateFitnessFunction() const;

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
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(sufficientStatistics, "sufficientStatistics");
  }

 private:
  //! The sufficient statistics for all points seen so far.  Each column
  //! corresponds to a category, and contains a count of each of the classes
  //! seen for points in that category.
  arma::Mat<size_t> sufficientStatistics;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "hoeffding_categorical_split_impl.hpp"

#endif
