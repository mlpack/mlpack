#ifndef KDE_DUAL_TREE_HPP
#define KDE_DUAL_TREE_HPP

#include <mlpack/core.h>
#include <mlpack/core/tree/spacetree.h>
#include <mlpack/core/tree/hrectbound.h>
#include <iostream>

namespace mlpack
{
  namespace kde
  {
    template <typename TKernel>
    class KdeDualTree
    {
      private:
        /* possibly, these refer to the same object */
        tree::BinarySpaceTree<bound::HRectBound<2> >* trainingRoot;
        tree::BinarySpaceTree<bound::HRectBound<2> >* queryRoot;
        arma::mat trainingData;
        arma::mat queryData;
        double bandwidth;
      public:
        /* the two data sets are different */
        KdeDualTree (arma::mat& trainingData, arma::mat& queryData);
        /* the training data is also the query data */
        KdeDualTree (arma::mat& trainingData);
        /* find a suitable bandwidth */
        double optimizeBandwidth (double lower, double upper, size_t attempts);
    };
  };
};

#include "kde_dual_tree_impl.hpp"

#endif
