
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
void KdeDualTree<TKernel, TTree>::MultiBandwidthDualTree()
{
  while (!nodePriorityQueue.empty())
  {
    /* get the first structure in the queue */
    struct queueNode queueCurrent = nodePriorityQueue.pop();
    TTree* Q = queueCurrent.Q;
    TTree* T = queueCurrent.T;
    size_t sizeOfTNode = T->count();
    size_t sizeOfQNode = Q->count();
    size_t QIndex = queueCurrent.QIndex;
    arma::vec deltaLower = queueCurrent.deltaLower;
    arma::vec deltaUpper = queueCurrent.deltaUpper;
    /* v is the level of the Q node */
    size_t v = GetLevelOfNode(Q);
    size_t bUpper = queueCurrent.bUpperIndex;
    size_t bLower = queueCurrent.bLowerIndex;
    /* check to see whether we've reached the epsilon condition */
    bool epsilonCondition = true;
    for (size_t bIndex = queueCurrent.bLowerIndex;
         bIndex <= queueCurrent.bUpperIndex;
         ++bIndex)
    {
      if (lowerBoundLevelByBandwidth(v,bIndex) > DBL_EPSILON)
      {
        double constraint = (upperBoundLevelByBandwidth(v,bIndex) -
                             lowerBoundLevelByBandwidth(v,bIndex)) /
                             lowerBoundLevelByBandwidth(v,bIndex);
        if (constraint > epsilon)
        {
          epsilonCondition = false;
          break;
        }
      }
      else
      {
        /* we haven't set this lower bound */
        epsilonCondition = false;
        break;
      }
    }
    /* return */
    if (epsilonCondition)
    {
      return;
    }
    /* we didn't meet the criteria; let's narrow the bandwidth range */
    Winnow(v, &bLower, &bUpper);
    if (queueCurrent.priority < PRIORITY_MAX)
    {
      double dMin = pow(Q->bound().MinDistance(T->bound()), 0.5);
      double dMax = pow(Q->bound().MaxDistance(T->bound()), 0.5);
      /* iterate through the remaining bandwidths */
      bool meetsDeltaCondition = true;
      std::vector<bool> deltaCondition;
      for (size_t bIndex = bLower; bIndex <= bUpper; ++bIndex)
      {
        double bandwidth = bandwidths[bIndex];
        double dl = sizeOfTNode * kernel(dMax / bandwidth);
        double du = sizeOfTNode * kernel(dMin / bandwidth);
        deltaLower(bIndex) = dl;
        deltaUpper(bIndex) = du - sizeOfTNode;
        if ((du - dl)/(lowerBoundQNodeByBandwidth(QIndex, bIndex) + dl) < delta)
        {
          for (size_t q = Q->begin(); q < Q->end(); ++q)
          {
            lowerBoundQPointByBandwidth(q,bIndex) += deltaLower(bIndex);
            upperBoundQPointByBandwidth(q,bIndex) += deltaUpper(bIndex);
          }
          /* subtract the current log-likelihood */
          upperBoundLevelByBandwidth(v, bIndex) -=
              sizeOfQNode * log(upperBoundQNodeByBandwidth(QIndex, bIndex));
          lowerBoundLevelByBandwidth(v, bIndex) -=
              sizeOfQNode * log(lowerBoundQNodeByBandwidth(QIndex, bIndex));
          /* adjust the current inner portion */
          lowerBoundQNodeByBandwidth(QIndex, bIndex) += deltaLower(bIndex);
          upperBoundQNodeByBandwidth(QIndex, bIndex) += deltaUpper(bIndex);
          /* add the corrected log-likelihood */
          upperBoundLevelByBandwidth(v, bIndex) +=
              sizeOfQNode * log(upperBoundQNodeByBandwidth(QIndex, bIndex));
          lowerBoundLevelByBandwidth(v, bIndex) +=
              sizeOfQNode * log(lowerBoundQNodeByBandwidth(QIndex, bIndex));
        }
        /* check the delta condition with the new values */
        if ((du - dl)/(lowerBoundQNodeByBandwidth(QIndex, bIndex) + dl) >= delta)
        {
          deltaCondition.push_back(false);
          meetsDeltaCondition = false;
        }
        else
        {
          deltaCondition.push_back(true);
        }
      }
      /* check whether we met the delta condition for
       *   all applicable bandwidths */
      if (meetsDeltaCondition)
      {
        /* adjust the current structure, then reinsert it into the queue */
        queueCurrent.dl = deltaLower;
        queueCurrent.du = deltaUpper;
        queueCurrent.bUpperIndex = bUpper;
        queueCurrent.bLowerIndex = bLower;
        queueCurrent.priority += PRIORITY_MAX;
        nodePriorityQueue.insert(queueCurrent);
        continue;
      }
      else
      {
        /* winnow according to the delta conditions */
        std::vector<bool>::iterator bIt = deltaCondition.begin();
        while (*bIt && bIt != deltaCondition.end())
        {
          ++bIt;
          ++bLower;
        }
        bIt = deltaCondition.end();
        --bIt;
        while (*bIt && bIt != deltaCondition.begin())
        {
          --bIt;
          --bUpper;
        }
      }
    }
    else /* the priority exceeds the maximum available */
    {
      deltaLower = -deltaLower;
      deltaUpper = -deltaUpper;
      for (size_t bIndex = bLower; bIndex <= bUpper; ++bIndex)
      {
        for (size_t q = Q->begin(); q < Q->end(); ++q)
        {
          lowerBoundQPointByBandwidth(q,bIndex) += deltaLower(bIndex);
          upperBoundQPointByBandwidth(q,bIndex) += deltaUpper(bIndex);
        }
        /* subtract the current log-likelihood */
        upperBoundLevelByBandwidth(v, bIndex) -=
            sizeOfQNode * log(upperBoundQNodeByBandwidth(QIndex, bIndex));
        lowerBoundLevelByBandwidth(v, bIndex) -=
            sizeOfQNode * log(lowerBoundQNodeByBandwidth(QIndex, bIndex));
        /* adjust the current inner portion */
        lowerBoundQNodeByBandwidth(QIndex, bIndex) += deltaLower(bIndex);
        upperBoundQNodeByBandwidth(QIndex, bIndex) += deltaUpper(bIndex);
        /* add the corrected log-likelihood */
        upperBoundLevelByBandwidth(v, bIndex) +=
            sizeOfQNode * log(upperBoundQNodeByBandwidth(QIndex, bIndex));
        lowerBoundLevelByBandwidth(v, bIndex) +=
            sizeOfQNode * log(lowerBoundQNodeByBandwidth(QIndex, bIndex));
      }
    }
    if (Q->is_leaf() && T->is_leaf())
    {
      MultiBandwidthDualTreeBase(Q, T, QIndex, bLower, bUpper);
    }
    double priority = pow(Q->bound().MinDistance(T->bound()), 0.5);
    if (!Q->is_left() && !T->is_leaf())
    {
      struct queueNode leftLeft =
      {T->left(),Q->left(), 2*QIndex + 1, arma::vec(deltaUpper),
        arma::vec(deltaLower), priority, bLower, bUpper};
      struct queueNode leftRight =
      {T->left(),Q->right(), 2*QIndex + 2, arma::vec(deltaUpper),
        arma::vec(deltaLower), priority, bLower, bUpper};
      struct queueNode rightLeft =
      {T->right(),Q->left(), 2*QIndex + 1, arma::vec(deltaUpper),
        arma::vec(deltaLower), priority, bLower, bUpper};
      struct queueNode rightRight =
      {T->right(),Q->right(), 2*QIndex + 2, arma::vec(deltaUpper),
        arma::vec(deltaLower), priority, bLower, bUpper};
      nodePriorityQueue.insert(leftLeft);
      nodePriorityQueue.insert(leftRight);
      nodePriorityQueue.insert(rightLeft);
      nodePriorityQueue.insert(rightRight);
    }
  }
}

void KdeDualTree<TKernel, TTree>::Winnow(size_t level,
                                         size_t* bLower,
                                         size_t* bUpper)
{
  size_t bIndex = *bLower;
  double constraint = delta;
  bool enteredTheLoop = false;
  /* bring the lower up */
  if (lowerBoundLevelByBandwidth(level,bIndex) > DBL_EPSILON)
  {
    constraint = (upperBoundLevelByBandwidth(level,bIndex) -
                  lowerBoundLevelByBandwidth(level,bIndex)) /
                  lowerBoundLevelByBandwidth(level,bIndex);
  }
  while (constraint < delta && bIndex <= *bUpper)
  {
    enteredTheLoop = true;
    ++bIndex;
    if (lowerBoundLevelByBandwidth(level,bIndex) > DBL_EPSILON)
    {
      constraint = (upperBoundLevelByBandwidth(level,bIndex) -
                    lowerBoundLevelByBandwidth(level,bIndex)) /
                    lowerBoundLevelByBandwidth(level,bIndex);
    }
    else
    {
      break;
    }
  }
  if (enteredTheLoop)
  {
    *bLower = bIndex - 1;
  }

  bIndex = *bUpper;
  constraint = delta;
  enteredTheLoop = false;
  /* bring the lower up */
  if (lowerBoundLevelByBandwidth(level,bIndex) > DBL_EPSILON)
  {
    constraint = (upperBoundLevelByBandwidth(level,bIndex) -
                  lowerBoundLevelByBandwidth(level,bIndex)) /
                  lowerBoundLevelByBandwidth(level,bIndex);
  }
  while (constraint < delta && bIndex >= *bLower)
  {
    enteredTheLoop = true;
    --bIndex;
    if (lowerBoundLevelByBandwidth(level,bIndex) > DBL_EPSILON)
    {
      constraint = (upperBoundLevelByBandwidth(level,bIndex) -
                    lowerBoundLevelByBandwidth(level,bIndex)) /
                    lowerBoundLevelByBandwidth(level,bIndex);
    }
    else
    {
      break;
    }
  }
  if (enteredTheLoop)
  {
    *bUpper = bIndex + 1;
  }
}


template<typename TKernel, typename TTree>
void KdeDualTree<TKernel, TTree>::MultiBandwidthDualTreeBase(TTree* Q,
                                TTree* T, size_t QIndex,
                                size_t lowerBIndex, size_t upperBIndex)
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
      size_t bandwidthIndex = upperBIndex;
      while (bandwidthIndex > lowerBIndex)
      {
        --bandwidthIndex;
        double bandwidth = bandwidths[bandwidthIndex];
        double scaledProduct = pow(distSquared, 0.5) / bandwidth;
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
    for (size_t bIndex = lowerBIndex; bIndex <= upperBIndex; ++bIndex)
    {
      upperBoundQPointByBandwidth(q, bIndex) -= sizeOfTNode;
    }
  }
  size_t levelOfQ = GetLevelOfNode(Q);
  for (size_t bIndex = lowerBIndex; bIndex <= upperBIndex; ++bIndex)
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
