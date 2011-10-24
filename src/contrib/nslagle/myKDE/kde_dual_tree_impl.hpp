
#ifndef KDE_DUAL_TREE_IMPL_HPP
#define KDE_DUAL_TREE_IMPL_HPP

#ifndef KDE_DUAL_TREE_HPP
#error "Do not include this header directly."
#endif

using namespace mlpack;
using namespace mlpack::kde;

namespace mlpack
{
namespace kde
{

template<typename TKernel>
KdeDualTree<TKernel>::KdeDualTree (arma::mat& train,
                                   arma::mat& query) :
  trainingRoot (new tree::BinarySpaceTree<bound::HRectBound<2> > (train)),
  queryRoot (new tree::BinarySpaceTree<bound::HRectBound<2> > (query))
{
  trainingData = train;
  queryData = query;
}
template<typename TKernel>
KdeDualTree<TKernel>::KdeDualTree (arma::mat& train) :
  trainingRoot (new tree::BinarySpaceTree<bound::HRectBound<2> > (train)),
  queryRoot (trainingRoot)
{
  trainingData = train;
  queryData = train;
}

};
};

#endif
