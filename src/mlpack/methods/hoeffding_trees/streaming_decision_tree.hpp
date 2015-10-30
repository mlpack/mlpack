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
                        const double confidence = 0.95,
                        const size_t numSamples = 5000,
                        const size_t checkInterval = 100);

  StreamingDecisionTree(const data::DatasetInfo& datasetInfo,
                        const size_t numClasses,
                        const double confidence = 0.95,
                        const size_t numSamples = 5000,
                        const size_t checkInterval = 100,
                        std::unordered_map<size_t, std::pair<size_t, size_t>>*
                            dimensionMappings = NULL);

  StreamingDecisionTree(const StreamingDecisionTree& other);

  size_t NumChildren() const { return children.size(); }
  StreamingDecisionTree& Child(const size_t i) { return children[i]; }
  const StreamingDecisionTree& Child(const size_t i) const { return children[i];
}

  const SplitType& Split() const { return split; }
  SplitType& Split() { return split; }

  template<typename VecType>
  void Train(const VecType& data, const size_t label);

  void Train(const MatType& data, const arma::Row<size_t>& labels);

  template<typename VecType>
  size_t Classify(const VecType& data);

  template<typename VecType>
  void Classify(const VecType& data, size_t& prediction, double& probability);

  void Classify(const MatType& data, arma::Row<size_t>& predictions);

  void Classify(const MatType& data,
                arma::Row<size_t>& predictions,
                arma::rowvec& probabilities);

  size_t& MajorityClass() { return split.MajorityClass(); }

  // How do we encode the actual split itself?

  // that's just a split dimension and a rule (categorical or numeric)

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(split, "split");
    ar & data::CreateNVP(checkInterval, "checkInterval");

    size_t numChildren;
    if (Archive::is_saving::value)
      numChildren = children.size();
    ar & data::CreateNVP(numChildren, "numChildren");
    if (Archive::is_loading::value)
      children.resize(numChildren, StreamingDecisionTree(data::DatasetInfo(), 0,
          0));

    for (size_t i = 0; i < numChildren; ++i)
    {
      std::ostringstream name;
      name << "child" << i;
      ar & data::CreateNVP(children[i], name.str());
    }
  }

 private:
  std::vector<StreamingDecisionTree> children;
  size_t checkInterval;

  SplitType split;
};

} // namespace tree
} // namespace mlpack

// Include implementation.
#include "streaming_decision_tree_impl.hpp"

// Include convenience typedefs.
#include "typedef.hpp"

#endif
