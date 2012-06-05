/**
 * @file max_ip.hpp
 * @author Ryan Curtin
 *
 * Definition of the MaxIP class, which is the maximum inner product search.
 */
#ifndef __MLPACK_METHODS_MAXIP_MAX_IP_HPP
#define __MLPACK_METHODS_MAXIP_MAX_IP_HPP

#include <mlpack/core.hpp>
#include "ip_metric.hpp"
#include <mlpack/core/tree/cover_tree.hpp>

namespace mlpack {
namespace maxip {

template<typename KernelType>
class MaxIP
{
 public:
  MaxIP(const arma::mat& referenceSet,
        KernelType& kernel,
        bool single = false,
        bool naive = false,
        double expansionConstant = 2.0);

  MaxIP(const arma::mat& referenceSet,
        const arma::mat& querySet,
        KernelType& kernel,
        bool single = false,
        bool naive = false,
        double expansionConstant = 2.0);

  ~MaxIP();

  void Search(const size_t k,
              arma::Mat<size_t>& indices,
              arma::mat& products);

 private:
  const arma::mat& referenceSet;

  const arma::mat& querySet;

  tree::CoverTree<IPMetric<KernelType> >* referenceTree;

  tree::CoverTree<IPMetric<KernelType> >* queryTree;

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

}; // namespace maxip
}; // namespace mlpack

// Include implementation.
#include "max_ip_impl.hpp"

#endif
