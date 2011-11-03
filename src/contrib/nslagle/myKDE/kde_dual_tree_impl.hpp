
#ifndef KDE_DUAL_TREE_IMPL_HPP
#define KDE_DUAL_TREE_IMPL_HPP

#ifndef KDE_DUAL_TREE_HPP
#ifndef USE_KDE_DUAL_TREE_IMPL_HPP
#error "Do not include this header directly."
#endif
#endif

using namespace mlpack;
using namespace mlpack::kde;

namespace mlpack
{
namespace kde
{

template<typename TKernel, typename TTree>
KdeDualTree<TKernel, TTree>::KdeDualTree (arma::mat& reference,
                                   arma::mat& query)
{
  referenceRoot (new TTree (reference)),
  queryRoot (new TTree (query))
  referenceData = reference;
  queryData = query;
  levelsInTree = queryRoot->levelsBelow();
  queryTreeSize = queryRoot->treeSize();
  SetDefaults();
}

template<typename TKernel, typename TTree>
KdeDualTree<TKernel, TTree>::KdeDualTree (arma::mat& reference)
{
  referenceData = reference;
  queryData = reference;

  referenceRoot = new TTree (reference, referenceShuffledIndices);
  queryRoot = referenceRoot;
  levelsInTree = queryRoot->levelsBelow();
  queryTreeSize = queryRoot->treeSize();
  SetDefaults();
}

template<typename TKernel, typename TTree>
KdeDualTree<TKernel, TTree>::SetDefaults()
{
  BandwidthRange(0.01, 100.0);
  bandwidthCount = 10;
  delta = epsilon = 0.05;
  kernel = TKernel(1.0);
}

template<typename TKernel, typename TTree>
void KdeDualTree<TKernel, TTree>::MultiBandwidthDualTreeBase(TTree* Q,
                                TTree* T, size_t QIndex,
                                std::set<double> remainingBandwidths)
{
  size_t sizeOfTNode = T->count();
  size_t sizeOfQNode = Q->count();
  for (size_t q = Q->begin(); q < Q->end(); ++q)
  {
    arma::vec queryPoint = queryData.unsafe_col(q);
    for (size_t t = T->begin(); t < T->end(); ++t)
    {
      arma::vec diff = queryPoint - referenceData.unsafe_col(t);
      double distSquared = arma::dot(diff, diff);
      std::set<double>::iterator bIt = remainingBandwidths.end();
      size_t bandwidthIndex = bandwidthCount;
      while (bIt != remainingBandwidths.begin())
      {
        --bIt;
        --bandwidthIndex;
        double bandwidth = *bIt;
        double scaledProduct = distSquared / (bandwidth * bandwidth);
        /* TODO: determine the power of the incoming argument */
        double contribution = kernel(scaledProduct);
        if (contribution > DBL_EPSILON)
        {
          upperBoundQPointByBandwidth(q, bandwidthIndex) += contribution;
          lowerBoundQPointByBandwidth(q, bandwidthIndex) += contribution;
        }
        else
        {
          break;
        }
      }
    }
    for (size_t bIndex = bandwidthCount - remainingBandwidths.size(); bIndex < remainingBandwidths.size(); ++bIndex)
    {
      upperBoundQPointByBandwidth(q, bIndex) -= sizeOfTNode;
    }
  }
  size_t levelOfQ = GetLevelOfNode(Q);
  for (size_t bIndex = bandwidthCount - remainingBandwidths.size(); bIndex < remainingBandwidths.size(); ++bIndex)
  {
    /* subtract out the current log-likelihood amount for this Q node so we can readjust
     *   the Q node bounds by current bandwidth */
    upperBoundLevelByBandwidth(levelOfQ, bIndex) -=
        sizeOfQNode * log(upperBoundQNodeByBandwidth(QIndex, bIndex));
    lowerBoundLevelByBandwidth(levelOfQ, bIndex) -=
        sizeOfQNode * log(lowerBoundQNodeByBandwidth(QIndex, bIndex));
    arma::vec upperBound = upperBoundQPointByBandwidth.unsafe_col(bIndex);
    arma::vec lowerBound = lowerBoundQPointByBandwidth.unsafe_col(bIndex);
    double minimumLower = lowerBoundQPointByBandwidth(Q->begin(), bIndex);
    double maximumUpper = upperBoundQPointByBandwidth(Q->begin(), bIndex);
    for (size_t q = Q->begin(); q < Q->end(); ++q)
    {
      if (lowerBoundQPointByBandwidth(q,bIndex) < minimumLower)
      {
        minimumLower = lowerBoundQPointByBandwidth(q,bIndex);
      }
      if (upperBoundQPointByBandwidth(q,bIndex) > maximumUpper)
      {
        maximumUpper = upperBoundQPointByBandwidth(q,bIndex);
      }
    }
    /* adjust Q node bounds, then add the new quantities to the level by bandwidth
     *   log-likelihood bounds */
    lowerBoundQNodeByBandwidth(QIndex, bIndex) = minimumLower;
    upperBoundQNodeByBandwidth(QIndex, bIndex) = maximumUpper - sizeOfTNode;
    upperBoundLevelByBandwidth(levelOfQ, bIndex) +=
        sizeOfQNode * log(upperBoundQNodeByBandwidth(QIndex, bIndex));
    lowerBoundLevelByBandwidth(levelOfQ, bIndex) +=
        sizeOfQNode * log(lowerBoundQNodeByBandwidth(QIndex, bIndex));
  }
}

};
};

#endif
