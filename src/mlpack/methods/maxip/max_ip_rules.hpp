/**
 * @file max_ip_rules.hpp
 * @author Ryan Curtin
 *
 * Rules for the single or dual tree traversal for the maximum inner product
 * search.
 */
#ifndef __MLPACK_METHODS_MAXIP_MAX_IP_RULES_HPP
#define __MLPACK_METHODS_MAXIP_MAX_IP_RULES_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

namespace mlpack {
namespace maxip {

template<typename MetricType>
class MaxIPRules
{
 public:
  MaxIPRules(const arma::mat& referenceSet,
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

}; // namespace maxip
}; // namespace mlpack

// Include implementation.
#include "max_ip_rules_impl.hpp"

#endif
