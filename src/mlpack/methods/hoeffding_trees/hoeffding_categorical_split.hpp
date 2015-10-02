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
 */
template<typename FitnessFunction>
class HoeffdingCategoricalSplit
{
 public:
  typedef CategoricalSplitInfo SplitInfo;

  HoeffdingCategoricalSplit(const size_t numCategories,
                            const size_t numClasses);

  template<typename eT>
  void Train(eT value, const size_t label);

  double EvaluateFitnessFunction() const;

  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);

  size_t MajorityClass() const;
  double MajorityProbability() const;

  //! Serialize the categorical split.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(sufficientStatistics, "sufficientStatistics");
  }

 private:
  arma::Mat<size_t> sufficientStatistics;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "hoeffding_categorical_split_impl.hpp"

#endif
