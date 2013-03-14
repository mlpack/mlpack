/**
 * @file fastmks.hpp
 * @author Ryan Curtin
 *
 * Definition of the FastMKS class, which is the fast max-kernel search.
 */
#ifndef __MLPACK_METHODS_FASTMKS_FASTMKS_HPP
#define __MLPACK_METHODS_FASTMKS_FASTMKS_HPP

#include <mlpack/core.hpp>
#include "ip_metric.hpp"
#include "fastmks_stat.hpp"
#include <mlpack/core/tree/cover_tree.hpp>

namespace mlpack {
namespace fastmks {

template<typename KernelType>
class FastMKS
{
 public:
  FastMKS(const arma::mat& referenceSet,
          KernelType& kernel,
          bool single = false,
          bool naive = false,
          double expansionConstant = 2.0);

  FastMKS(const arma::mat& referenceSet,
          const arma::mat& querySet,
          KernelType& kernel,
          bool single = false,
          bool naive = false,
          double expansionConstant = 2.0);

  ~FastMKS();

  void Search(const size_t k,
              arma::Mat<size_t>& indices,
              arma::mat& products);

 private:
  const arma::mat& referenceSet;

  const arma::mat& querySet;

  tree::CoverTree<IPMetric<KernelType>, tree::FirstPointIsRoot, FastMKSStat>*
      referenceTree;

  tree::CoverTree<IPMetric<KernelType>, tree::FirstPointIsRoot, FastMKSStat>*
      queryTree;

  bool single;

  bool naive;

  IPMetric<KernelType> metric;

  // Utility function.  Copied too many times from too many places.
  void InsertNeighbor(arma::Mat<size_t>& indices,
                      arma::mat& products,
                      const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "fastmks_impl.hpp"

#endif
