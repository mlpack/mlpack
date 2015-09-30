/**
 * @file streaming_decision_tree.hpp
 * @author Ryan Curtin
 *
 * The core class for a streaming decision tree.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_STREAMING_DECISION_TREE_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_STREAMING_DECISION_TREE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

template<
  typename SplitType,
  typename MatType = arma::mat
>
class StreamingDecisionTree
{
 public:
  StreamingDecisionTree(const MatType& data,
                        const data::DatasetInfo& datasetInfo,
                        const arma::Row<size_t>& labels,
                        const size_t numClasses,
                        const double confidence = 0.95);

  StreamingDecisionTree(const data::DatasetInfo& datasetInfo,
                        const size_t dimensionality,
                        const size_t numClasses,
                        const double confidence = 0.95);

  StreamingDecisionTree(const StreamingDecisionTree& other);

  size_t NumChildren() const { return children.size(); }
  StreamingDecisionTree& Child(const size_t i) { return children[i]; }
  const StreamingDecisionTree& Child(const size_t i) const { return children[i];
}

  const SplitType& Split() const { return split; }

  template<typename VecType>
  void Train(const VecType& data, const size_t label);

  void Train(const MatType& data, const arma::Row<size_t>& labels);

  template<typename VecType>
  size_t Classify(const VecType& data);

  void Classify(const MatType& data, arma::Row<size_t>& predictions);

  size_t& MajorityClass() { return split.MajorityClass(); }

  // How do we encode the actual split itself?

  // that's just a split dimension and a rule (categorical or numeric)

 private:
  std::vector<StreamingDecisionTree> children;

  SplitType split;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "streaming_decision_tree_impl.hpp"

#endif
