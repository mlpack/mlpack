#ifndef KDE_DUAL_TREE_HPP
#define KDE_DUAL_TREE_HPP

#include <iostream>
#include <priority_queue>

#include <mlpack/core.h>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/hrectbound.hpp>
#include <mlpack/core/math/range.hpp>

#define PRIORITY_MAX DBL_MAX

namespace mlpack
{
namespace kde
{
/* structure within the priority queue */
struct queueNode
{
  TTree* T;
  TTree* Q;
  size_t QIndex;
  arma::vec deltaLower;
  arma::vec deltaUpper;
  double priority;
  size_t bLowerIndex;
  size_t bUpperIndex;
};
class QueueNodeCompare
{
  bool reverse;
 public:
  QueueNodeCompare(const bool& revparam=false) : reverse(revparam) {}
  bool operator() (const struct queueNode& lhs,
                   const struct queueNode& rhs) const
  {
    if (reverse)
      return (lhs.priority>rhs.priority);
    else
      return (lhs.priority<rhs.priority);
  }
};

template <typename TKernel = kernel::GaussianKernel,
          typename TTree = tree::BinarySpaceTree<bound::HRectBound<2> > >
class KdeDualTree
{
 private:
  TKernel kernel;
  /* possibly, these refer to the same object */
  TTree* referenceRoot;
  TTree* queryRoot;
  std::vector<size_t> referenceShuffledIndices;
  std::vector<size_t> queryShuffledIndices;
  arma::mat referenceData;
  arma::mat queryData;
  arma::mat upperBoundLevelByBandwidth;
  arma::mat lowerBoundLevelByBandwidth;
  arma::mat upperBoundQPointByBandwidth;
  arma::mat lowerBoundQPointByBandwidth;
  arma::mat upperBoundQNodeByBandwidth;
  arma::mat lowerBoundQNodeByBandwidth;
  /* relative estimate to limit bandwidth calculations */
  double delta;
  /* relative error with respect to the density estimate */
  double epsilon;
  math::Range bandwidths;
  std::priority_queue<struct queueNode,
                      std::vector<struct queueNode>,
                      QueueNodeCompare> nodePriorityQueue;
  size_t bandwidthCount;
  std::vector<double> bandwidths;
  size_t levelsInTree;
  size_t queryTreeSize;

  void SetDefaults();
  void MultiBandwidthDualTree();
  void MultiBandwidthDualTreeBase(TTree* Q,
                                  TTree* T, size_t QIndex,
                                  size_t lowerBIndex, size_t upperBIndex);
  double GetPriority(TTree* nodeQ, TTree* nodeT)
  {
    return nodeQ->bound().MinDistance(*nodeT);
  }
  size_t GetLevelOfNode(TTree* node)
  {
    return levelsInTree - node->levelsBelow();
  }
  void Winnow(size_t bLower, size_t bUpper, size_t* newLower, size_t* newUpper);
 public:
  /* the two data sets are different */
  KdeDualTree (arma::mat& referenceData, arma::mat& queryData);
  /* the reference data is also the query data */
  KdeDualTree (arma::mat& referenceData);
  /* setters and getters */
  const math::Range& BandwidthRange() const { return bandwidthRange; }
  const size_t& BandwidthCount() const { return bandwidthCount; }
  const double& Delta() const { return delta; }
  const double& Epsilon() const { return epsilon; }

  void BandwidthRange(double l, double u) { bandwidthRange = math::Range(l,u); }
  size_t& BandwidthCount() { return bandwidthCount; }
  double& Delta() { return delta; }
  double& Epsilon() { return epsilon; }
};
}; /* end namespace kde */
}; /* end namespace mlpack */

#define USE_KDE_DUAL_TREE_IMPL_HPP
#include "kde_dual_tree_impl.hpp"
#undef USE_KDE_DUAL_TREE_IMPL_HPP

#endif
