
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
}

template<typename TKernel, typename TTree>
void KdeDualTree<TKernel, TTree>::MultiBandwidthDualTreeBase(TTree* Q,
                                TTree* T,
                                std::set<double> remainingBandwidths)
{
  for (size_t q = Q->begin(); q < Q->end(); ++q)
  {
    for (size_t t = T->begin(); t < T->end(); ++t)
    {
    }
  }
}


};
};

#endif
