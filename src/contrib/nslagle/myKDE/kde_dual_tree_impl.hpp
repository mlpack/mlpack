
#ifndef KDE_DUAL_TREE_IMPL_HPP
#define KDE_DUAL_TREE_IMPL_HPP

#ifndef KDE_DUAL_TREE_HPP
#ifndef USE_KDE_DUAL_TREE_IMPL_HPP
#error "Do not include this header directly."
#endif
#endif

#define MADEIT std::cout<<"made it to "<<__LINE__<<" in "<<__FILE__<<std::endl

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
  referenceRoot = new TTree (reference, referenceShuffledIndices),
  queryRoot = new TTree (query, queryShuffledIndices);
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
  queryShuffledIndices = referenceShuffledIndices;
  levelsInTree = queryRoot->levelsBelow();
  queryTreeSize = queryRoot->treeSize();
  SetDefaults();
}

template<typename TKernel, typename TTree>
void KdeDualTree<TKernel, TTree>::SetDefaults()
{
  SetBandwidthBounds(0.01, 100.0);
  bandwidthCount = 10;
  delta = epsilon = 0.05;
  kernel = TKernel(1.0);
  nextAvailableNodeIndex = 0;
}

template<typename TKernel, typename TTree>
std::vector<double> KdeDualTree<TKernel, TTree>::Calculate()
{
  /* calculate the bandwidths */
  bandwidths.clear();
  inverseBandwidths.clear();

  if (bandwidthCount > 1)
  {
    double bandwidthDelta = (highBandwidth - lowBandwidth) / (bandwidthCount - 1);
    for (size_t bIndex = 0; bIndex < bandwidthCount; ++bIndex)
    {
      bandwidths.push_back(lowBandwidth + bandwidthDelta * bIndex);
      inverseBandwidths.push_back(1.0 / bandwidths.back());
    }
  }
  else
  {
    bandwidths.push_back(lowBandwidth);
    inverseBandwidths.push_back(1.0 / lowBandwidth);
  }

  /* resize the critical matrices */
  upperBoundLevelByBandwidth.zeros(levelsInTree,bandwidthCount);
  for (size_t bIndex = 0; bIndex < bandwidthCount; ++bIndex)
  {
    arma::vec col = upperBoundLevelByBandwidth.unsafe_col(bIndex);
    col.fill(referenceRoot->count() * inverseBandwidths[bIndex]);
  }
  upperBoundLevelByBandwidth.fill(referenceRoot->count());
  lowerBoundLevelByBandwidth.zeros(levelsInTree,bandwidthCount);
  upperBoundQPointByBandwidth.zeros(queryRoot->count(),bandwidthCount);
  for (size_t bIndex = 0; bIndex < bandwidthCount; ++bIndex)
  {
    arma::vec col = upperBoundQPointByBandwidth.unsafe_col(bIndex);
    col.fill(referenceRoot->count() * inverseBandwidths[bIndex]);
  }
  lowerBoundQPointByBandwidth.zeros(queryRoot->count(),bandwidthCount);
  upperBoundQNodeByBandwidth.zeros(queryTreeSize,bandwidthCount);
  for (size_t bIndex = 0; bIndex < bandwidthCount; ++bIndex)
  {
    arma::vec col = upperBoundQNodeByBandwidth.unsafe_col(bIndex);
    col.fill(referenceRoot->count() * inverseBandwidths[bIndex]);
  }
  lowerBoundQNodeByBandwidth.zeros(queryTreeSize,bandwidthCount);

  arma::vec dl;
  arma::vec du;
  dl.zeros(bandwidthCount);
  du.zeros(bandwidthCount);
  double priority = pow(
      queryRoot->bound().MinDistance(referenceRoot->bound()),
      0.5);
  struct queueNode<TTree> firstNode =
      {referenceRoot,queryRoot, nextAvailableNodeIndex, dl, du,
        priority, 0, bandwidthCount - 1};
  nodeIndices[queryRoot] = nextAvailableNodeIndex;
  ++nextAvailableNodeIndex;
  nodePriorityQueue.push(firstNode);
  size_t finalLevel = MultiBandwidthDualTree();

  size_t maxIndex = 0;
  double maxLogLikelihood = (upperBoundLevelByBandwidth(finalLevel,0) +
                             lowerBoundLevelByBandwidth(finalLevel,0)) / 2.0;
  for (size_t bIndex = 1; bIndex < bandwidthCount; ++bIndex)
  {
    double currentLogLikelihood = (upperBoundLevelByBandwidth(finalLevel,bIndex) +
                                   lowerBoundLevelByBandwidth(finalLevel,bIndex)) / 2.0;
    if (currentLogLikelihood > maxLogLikelihood)
    {
      currentLogLikelihood = maxLogLikelihood;
      maxIndex = bIndex;
    }
  }
  std::cout << upperBoundLevelByBandwidth << "\n";
  std::cout << lowerBoundLevelByBandwidth << "\n";
  std::cout << "best bandwidth " << bandwidths[maxIndex] << ";\n";
  exit(1);
  std::vector<double> densities;
  for (std::vector<size_t>::iterator shuffIt = queryShuffledIndices.begin();
      shuffIt != queryShuffledIndices.end(); ++shuffIt)
  {
    densities.push_back((upperBoundQPointByBandwidth(*shuffIt, maxIndex) +
                         lowerBoundQPointByBandwidth(*shuffIt, maxIndex)) / (2.0 * referenceRoot->count()));

  }
  return densities;
}

template<typename TKernel, typename TTree>
size_t KdeDualTree<TKernel, TTree>::MultiBandwidthDualTree()
{
  /* current level */
  size_t v = 0;
  while (!nodePriorityQueue.empty())
  {
    /* get the first structure in the queue */
    struct queueNode<TTree> queueCurrent = nodePriorityQueue.top();
    nodePriorityQueue.pop();
    TTree* Q = queueCurrent.Q;
    TTree* T = queueCurrent.T;
    size_t sizeOfTNode = T->count();
    size_t sizeOfQNode = Q->count();
    size_t QIndex = queueCurrent.QIndex;
    arma::vec deltaLower = queueCurrent.deltaLower;
    arma::vec deltaUpper = queueCurrent.deltaUpper;
    /* v is the level of the Q node */
    v = GetLevelOfNode(Q);
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
      return v;
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
        double inverseBandwidth = inverseBandwidths[bIndex];
        double dl = sizeOfTNode * inverseBandwidth * kernel.Evaluate(dMax * inverseBandwidth);
        double du = sizeOfTNode * inverseBandwidth * kernel.Evaluate(dMin * inverseBandwidth);
        deltaLower(bIndex) = dl;
        deltaUpper(bIndex) = du - sizeOfTNode;
        //std::cout << "QIndex: " << QIndex << " bIndex: " << bIndex << std::endl;
        //std::cout << "max QIndex: " << queryTreeSize - 1 << std::endl;
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
        queueCurrent.deltaLower = deltaLower;
        queueCurrent.deltaUpper = deltaUpper;
        queueCurrent.bUpperIndex = bUpper;
        queueCurrent.bLowerIndex = bLower;
        queueCurrent.priority += PRIORITY_MAX;
        nodePriorityQueue.push(queueCurrent);
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
    if (!Q->is_leaf() && !T->is_leaf())
    {
      //std::cout << "QIndex for the current non-leaf : " << QIndex << std::endl;
      TTree* QLeft = Q->left();
      TTree* QRight = Q->right();
      if (nodeIndices.find(QLeft) == nodeIndices.end())
      {
        nodeIndices[QLeft] = nextAvailableNodeIndex;
        ++nextAvailableNodeIndex;
      }
      if (nodeIndices.find(QRight) == nodeIndices.end())
      {
        nodeIndices[QRight] = nextAvailableNodeIndex;
        ++nextAvailableNodeIndex;
      }
      size_t QLeftIndex = (*(nodeIndices.find(QLeft))).second;
      size_t QRightIndex = (*(nodeIndices.find(QRight))).second;
      struct queueNode<TTree> leftLeft =
      {T->left(),Q->left(), QLeftIndex, arma::vec(deltaLower),
        arma::vec(deltaUpper), priority, bLower, bUpper};
      struct queueNode<TTree> leftRight =
      {T->left(),Q->right(), QRightIndex, arma::vec(deltaLower),
        arma::vec(deltaUpper), priority, bLower, bUpper};
      struct queueNode<TTree> rightLeft =
      {T->right(),Q->left(), QLeftIndex, arma::vec(deltaLower),
        arma::vec(deltaUpper), priority, bLower, bUpper};
      struct queueNode<TTree> rightRight =
      {T->right(),Q->right(), QRightIndex, arma::vec(deltaLower),
        arma::vec(deltaUpper), priority, bLower, bUpper};
      nodePriorityQueue.push(leftLeft);
      nodePriorityQueue.push(leftRight);
      nodePriorityQueue.push(rightLeft);
      nodePriorityQueue.push(rightRight);
    }
  }
  return v;
}

template<typename TKernel, typename TTree>
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
      size_t bandwidthIndex = upperBIndex + 1;
      while (bandwidthIndex > lowerBIndex)
      {
        --bandwidthIndex;
        double inverseBandwidth = inverseBandwidths[bandwidthIndex];
        double scaledProduct = pow(distSquared, 0.5) * inverseBandwidth;
        /* TODO: determine the power of the incoming argument */
        double contribution = inverseBandwidth * kernel.Evaluate(scaledProduct);
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
template<typename TKernel, typename TTree>
void KdeDualTree<TKernel, TTree>::SetBandwidthBounds(double l, double u)
{
  if (u <= l + DBL_EPSILON || l <= DBL_EPSILON)
  {
    Log::Fatal << "Incorrect bandwidth range assignment" << std::endl;
  }
  lowBandwidth = l;
  highBandwidth = u;
}

};
};

#endif
