/**
 * @file fastmks_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for the single or dual tree traversal for fast max-kernel search.
 */
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_RULES_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree/cover_tree.hpp>

namespace mlpack {
namespace fastmks {

template<typename MetricType>
class FastMKSRules
{
 public:
  FastMKSRules(const arma::mat& referenceSet,
               const arma::mat& querySet,
               arma::Mat<size_t>& indices,
               arma::mat& products);

  void BaseCase(const size_t queryIndex, const size_t referenceIndex);

  bool CanPrune(const size_t queryIndex,
                tree::CoverTree<MetricType>& referenceNode,
                const size_t parentIndex);

 private:
  const arma::mat& referenceSet;

  const arma::mat& querySet;

  arma::Mat<size_t>& indices;

  arma::mat& products;

  arma::vec queryKernels; // || q || for each q.

  void InsertNeighbor(const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "fastmks_rules_impl.hpp"

#endif
