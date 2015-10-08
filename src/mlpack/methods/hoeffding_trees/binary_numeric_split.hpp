/**
 * @file binary_numeric_split.hpp
 * @author Ryan Curtin
 *
 * An implementation of the binary-tree-based numeric splitting procedure
 * described by Gama, Rocha, and Medas in their KDD 2003 paper.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_SPLIT_BINARY_NUMERIC_SPLIT_HPP
#define __MLPACK_METHODS_HOEFFDING_SPLIT_BINARY_NUMERIC_SPLIT_HPP

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
 */
template<typename FitnessFunction,
         typename ObservationType = double>
class BinaryNumericSplit
{
 public:
  typedef NumericSplitInfo<ObservationType> SplitInfo;

  BinaryNumericSplit(const size_t numClasses);

  void Train(ObservationType value, const size_t label);

  double EvaluateFitnessFunction();

  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);

  size_t MajorityClass() const;
  double MajorityProbability() const;

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  // Return the number of children if this node were to split on this feature.
  size_t NumChildren() const { return 2; }

 private:
  // All we need is ordered access.
  std::multimap<ObservationType, size_t> sortedElements;

  arma::Col<size_t> classCounts;

  bool isAccurate;
  ObservationType bestSplit;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "binary_numeric_split_impl.hpp"

#endif
