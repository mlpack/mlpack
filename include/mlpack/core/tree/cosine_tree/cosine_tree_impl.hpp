/**
 * @file core/tree/cosine_tree/cosine_tree_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of cosine tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP

#include "cosine_tree.hpp"

namespace mlpack {

template<typename MatType>
inline CosineTree<MatType>::CosineTree(const MatType& dataset) :
    dataset(&dataset),
    parent(NULL),
    left(NULL),
    right(NULL),
    numColumns(dataset.n_cols),
    localDataset(false)
{
  // Initialize sizes of column indices and l2 norms.
  indices.resize(numColumns);
  l2NormsSquared.zeros(numColumns);

  // Set indices and calculate squared norms of the columns.
  for (size_t i = 0; i < numColumns; ++i)
  {
    indices[i] = i;
    double l2Norm = (double) arma::norm(dataset.col(i), 2);
    l2NormsSquared(i) = l2Norm * l2Norm;
  }

  // Frobenius norm of columns in the node.
  frobNormSquared = accu(l2NormsSquared);

  // Calculate centroid of columns in the node.
  CalculateCentroid();

  splitPointIndex = ColumnSampleLS();
}

template<typename MatType>
inline CosineTree<MatType>::CosineTree(CosineTree& parentNode,
                                       const std::vector<size_t>& subIndices) :
    dataset(&parentNode.GetDataset()),
    parent(&parentNode),
    left(NULL),
    right(NULL),
    numColumns(subIndices.size()),
    localDataset(false)
{
  // Initialize sizes of column indices and l2 norms.
  indices.resize(numColumns);
  l2NormsSquared.zeros(numColumns);

  // Set indices and squared norms of the columns.
  for (size_t i = 0; i < numColumns; ++i)
  {
    indices[i] = parentNode.indices[subIndices[i]];
    l2NormsSquared(i) = parentNode.l2NormsSquared(subIndices[i]);
  }

  // Frobenius norm of columns in the node.
  frobNormSquared = accu(l2NormsSquared);

  // Calculate centroid of columns in the node.
  CalculateCentroid();

  splitPointIndex = ColumnSampleLS();
}

template<typename MatType>
inline CosineTree<MatType>::CosineTree(const MatType& dataset,
                                       const double epsilon,
                                       const double delta) :
    dataset(&dataset),
    delta(delta),
    left(NULL),
    right(NULL),
    localDataset(false)
{
  // Declare the cosine tree priority queue.
  CosineNodeQueue<MatType> treeQueue;
  CompareCosineNode comp;

  // Define root node of the tree and add it to the queue.
  CosineTree root(dataset);
  VecType tempVector = arma::zeros<VecType>(dataset.n_rows);
  root.L2Error(-1.0); // We don't know what the error is.
  root.BasisVector(tempVector);
  treeQueue.push_back(&root);
  // treeQueue is empty now, so we don't need to call std::push_heap here.

  // Initialize Monte Carlo error estimate for comparison.
  double monteCarloError = root.FrobNormSquared();

  while (treeQueue.size() > 0 &&
         (monteCarloError > epsilon * root.FrobNormSquared()))
  {
    // Pop node from queue with highest projection error.
    CosineTree* currentNode;
    currentNode = treeQueue.front();
    std::pop_heap(treeQueue.begin(), treeQueue.end(), comp);
    treeQueue.pop_back();

    // If the priority is 0, we can't improve anything, and we can assume that
    // we've done the best we can.
    if (currentNode->L2Error() == 0.0)
    {
      Log::Warn << "CosineTree::CosineTree(): could not build tree to "
          << "desired relative error " << epsilon << "; failing with estimated "
          << "relative error " << (monteCarloError / root.FrobNormSquared())
          << "." << std::endl;
      break;
    }

    // Split the node into left and right children.  We assume that this cannot
    // fail; it might fail if L2Error() is 0, but we have already avoided that
    // case.
    currentNode->CosineNodeSplit();

    // Obtain pointers to the left and right children of the current node.
    CosineTree *currentLeft, *currentRight;
    currentLeft = currentNode->Left();
    currentRight = currentNode->Right();

    // Calculate basis vectors of left and right children.
    VecType lBasisVector, rBasisVector;

    ModifiedGramSchmidt(treeQueue, currentLeft->Centroid(), lBasisVector);
    ModifiedGramSchmidt(treeQueue, currentRight->Centroid(), rBasisVector,
                        &lBasisVector);

    // Add basis vectors to their respective nodes.
    currentLeft->BasisVector(lBasisVector);
    currentRight->BasisVector(rBasisVector);

    // Calculate Monte Carlo error estimates for child nodes.
    MonteCarloError(currentLeft, treeQueue, &lBasisVector, &rBasisVector);
    MonteCarloError(currentRight, treeQueue, &lBasisVector, &rBasisVector);

    // Push child nodes into the priority queue.
    treeQueue.push_back(currentLeft);
    std::push_heap(treeQueue.begin(), treeQueue.end(), comp);
    treeQueue.push_back(currentRight);
    std::push_heap(treeQueue.begin(), treeQueue.end(), comp);

    // Calculate Monte Carlo error estimate for the root node.
    monteCarloError = MonteCarloError(&root, treeQueue);
  }

  // Construct the subspace basis from the current priority queue.
  ConstructBasis(treeQueue);
}

//! Copy the given tree.
template<typename MatType>
inline CosineTree<MatType>::CosineTree(const CosineTree& other) :
    // Copy matrix, but only if we are the root.
    dataset((other.parent == NULL) ? new MatType(*other.dataset) : NULL),
    delta(other.delta),
    parent(NULL),
    left(NULL),
    right(NULL),
    indices(other.indices),
    l2NormsSquared(other.l2NormsSquared),
    centroid(other.centroid),
    basisVector(other.basisVector),
    splitPointIndex(other.SplitPointIndex()),
    numColumns(other.NumColumns()),
    l2Error(other.L2Error()),
    frobNormSquared(other.FrobNormSquared()),
    localDataset(other.parent == NULL)
{
  // Create left and right children (if any).
  if (other.Left())
  {
    left = new CosineTree(*other.Left());
    left->Parent() = this; // Set parent to this, not other tree.
  }

  if (other.Right())
  {
    right = new CosineTree(*other.Right());
    right->Parent() = this; // Set parent to this, not other tree.
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL && localDataset)
  {
    std::queue<CosineTree*> queue;
    if (left)
      queue.push(left);
    if (right)
      queue.push(right);
    while (!queue.empty())
    {
      CosineTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      if (node->left)
        queue.push(node->left);
      if (node->right)
        queue.push(node->right);
    }
  }
}

//! Copy assignment operator: copy the given other tree.
template<typename MatType>
inline CosineTree<MatType>& CosineTree<MatType>::operator=(
    const CosineTree& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  if (localDataset)
    delete dataset;

  delete left;
  delete right;

  // Performing a deep copy of the dataset.
  dataset = (other.parent == NULL) ? new MatType(*other.dataset) : NULL;

  delta = other.delta;
  parent = other.Parent();
  left = other.Left();
  right = other.Right();
  indices = other.indices;
  l2NormsSquared = other.l2NormsSquared;
  centroid = other.centroid;
  basisVector = other.basisVector;
  splitPointIndex = other.SplitPointIndex();
  numColumns = other.NumColumns();
  l2Error = other.L2Error();
  localDataset = (other.parent == NULL) ? true : false;
  frobNormSquared = other.FrobNormSquared();

  // Create left and right children (if any).
  if (other.Left())
  {
    left = new CosineTree(*other.Left());
    left->Parent() = this; // Set parent to this, not other tree.
  }

  if (other.Right())
  {
    right = new CosineTree(*other.Right());
    right->Parent() = this; // Set parent to this, not other tree.
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL && localDataset)
  {
    std::queue<CosineTree*> queue;
    if (left)
      queue.push(left);
    if (right)
      queue.push(right);
    while (!queue.empty())
    {
      CosineTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      if (node->left)
        queue.push(node->left);
      if (node->right)
        queue.push(node->right);
    }
  }

  return *this;
}

//! Move the given tree.
template<typename MatType>
inline CosineTree<MatType>::CosineTree(CosineTree&& other) :
    dataset(other.dataset),
    delta(std::move(other.delta)),
    parent(other.parent),
    left(other.left),
    right(other.right),
    indices(std::move(other.indices)),
    l2NormsSquared(std::move(other.l2NormsSquared)),
    centroid(std::move(other.centroid)),
    basisVector(std::move(other.basisVector)),
    splitPointIndex(other.splitPointIndex),
    numColumns(other.numColumns),
    l2Error(other.l2Error),
    frobNormSquared(other.frobNormSquared),
    localDataset(other.localDataset)
{
  // Now we are a clone of the other tree.  But we must also clear the other
  // tree's contents, so it doesn't delete anything when it is destructed.
  other.dataset = NULL;
  other.parent = NULL;
  other.left = NULL;
  other.right = NULL;
  other.splitPointIndex = 0;
  other.numColumns = 0;
  other.l2Error = -1;
  other.localDataset = false;
  other.frobNormSquared = 0;
  // Set new parent.
  if (left)
    left->parent = this;
  if (right)
    right->parent = this;
}

//! Move assignment operator: take ownership of the given tree.
template<typename MatType>
inline CosineTree<MatType>& CosineTree<MatType>::operator=(CosineTree&& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  if (localDataset)
    delete dataset;
  delete left;
  delete right;

  dataset = other.dataset;
  delta = std::move(other.delta);
  parent = other.Parent();
  left = other.Left();
  right = other.Right();
  indices = std::move(other.indices);
  l2NormsSquared = std::move(other.l2NormsSquared);
  centroid = std::move(other.centroid);
  basisVector = std::move(other.basisVector);
  splitPointIndex = other.SplitPointIndex();
  numColumns = other.NumColumns();
  l2Error = other.L2Error();
  localDataset = other.localDataset;
  frobNormSquared = other.FrobNormSquared();

  // Now we are a clone of the other tree.  But we must also clear the other
  // tree's contents, so it doesn't delete anything when it is destructed.
  other.dataset = NULL;
  other.parent = NULL;
  other.left = NULL;
  other.right = NULL;
  other.splitPointIndex = 0;
  other.numColumns = 0;
  other.l2Error = -1;
  other.localDataset = false;
  other.frobNormSquared = 0;
  // Set new parent.
  if (left)
    left->parent = this;
  if (right)
    right->parent = this;

  return *this;
}

template<typename MatType>
inline CosineTree<MatType>::~CosineTree()
{
  if (localDataset)
    delete dataset;
  if (left)
    delete left;
  if (right)
    delete right;
}

template<typename MatType>
inline void CosineTree<MatType>::ModifiedGramSchmidt(
    CosineNodeQueue<MatType>& treeQueue,
    typename CosineTree<MatType>::VecType& centroid,
    typename CosineTree<MatType>::VecType& newBasisVector,
    typename CosineTree<MatType>::VecType* addBasisVector)
{
  // Set new basis vector to centroid.
  newBasisVector = centroid;

  // Variables for iterating throught the priority queue.
  CosineTree *currentNode;
  typename CosineNodeQueue<MatType>::const_iterator i = treeQueue.cbegin();

  // For every vector in the current basis, remove its projection from the
  // centroid.
  for ( ; i != treeQueue.cend(); ++i)
  {
    currentNode = *i;

    double projection = (double) dot(currentNode->BasisVector(), centroid);
    newBasisVector -= projection * currentNode->BasisVector();
  }

  // If additional basis vector is passed, take it into account.
  if (addBasisVector)
  {
    double projection = (double) dot(*addBasisVector, centroid);
    newBasisVector -= *addBasisVector * projection;
  }

  // Normalize the modified centroid vector.
  if (arma::norm(newBasisVector, 2))
    newBasisVector /= arma::norm(newBasisVector, 2);
}

template<typename MatType>
inline double CosineTree<MatType>::MonteCarloError(
    CosineTree* node,
    CosineNodeQueue<MatType>& treeQueue,
    typename CosineTree<MatType>::VecType* addBasisVector1,
    typename CosineTree<MatType>::VecType* addBasisVector2)
{
  std::vector<size_t> sampledIndices;
  VecType probabilities;

  // Sample O(log m) points from the input node's distribution.
  // 'm' is the number of columns present in the node.
  size_t numSamples = std::log(node->NumColumns()) + 1;
  node->ColumnSamplesLS(sampledIndices, probabilities, numSamples);

  // Get pointer to the original dataset.
  const MatType& dataset = node->GetDataset();

  // Initialize weighted projection magnitudes as zeros.
  VecType weightedMagnitudes;
  weightedMagnitudes.zeros(numSamples);

  // Set size of projection vector, depending on whether additional basis
  // vectors are passed.
  size_t projectionSize;
  if (addBasisVector1 && addBasisVector2)
    projectionSize = treeQueue.size() + 2;
  else
    projectionSize = treeQueue.size();

  // For each sample, calculate the weighted projection onto the current basis.
  for (size_t i = 0; i < numSamples; ++i)
  {
    // Initialize projection as a vector of zeros.
    VecType projection;
    projection.zeros(projectionSize);

    CosineTree *currentNode;
    typename CosineNodeQueue<MatType>::const_iterator j = treeQueue.cbegin();

    size_t k = 0;
    // Compute the projection of the sampled vector onto the existing subspace.
    for ( ; j != treeQueue.cend(); ++j, ++k)
    {
      currentNode = *j;

      projection(k) = dot(dataset.col(sampledIndices[i]),
                                currentNode->BasisVector());
    }
    // If two additional vectors are passed, take their projections.
    if (addBasisVector1 && addBasisVector2)
    {
      projection(k++) = dot(dataset.col(sampledIndices[i]),
                                  *addBasisVector1);
      projection(k) = dot(dataset.col(sampledIndices[i]),
                                *addBasisVector2);
    }

    // Calculate the Frobenius norm squared of the projected vector.
    double frobProjection = arma::norm(projection, "frob");
    double frobProjectionSquared = frobProjection * frobProjection;

    // Calculate the weighted projection magnitude.
    weightedMagnitudes(i) = frobProjectionSquared / probabilities(i);
  }

  // Compute mean and standard deviation of the weighted samples.
  double mu = arma::mean(weightedMagnitudes);
  double sigma = arma::stddev(weightedMagnitudes);

  if (!sigma)
  {
    node->L2Error(node->FrobNormSquared() - mu);
    return (node->FrobNormSquared() - mu);
  }

  // Fit a normal distribution using the calculated statistics, and calculate a
  // lower bound on the magnitudes for the passed 'delta' parameter.
  double lowerBound = Quantile(delta, mu, sigma);

  // Upper bound on the subspace reconstruction error.
  node->L2Error(node->FrobNormSquared() - lowerBound);

  return (node->FrobNormSquared() - lowerBound);
}

template<typename MatType>
inline void CosineTree<MatType>::ConstructBasis(
    CosineNodeQueue<MatType>& treeQueue)
{
  // Initialize basis as matrix of zeros.
  basis.zeros(dataset->n_rows, treeQueue.size());

  // Variables for iterating through the priority queue.
  CosineTree *currentNode;
  typename CosineNodeQueue<MatType>::const_iterator i = treeQueue.cbegin();

  // Transfer basis vectors from the queue to the basis matrix.
  size_t j = 0;
  for ( ; i != treeQueue.cend(); ++i, ++j)
  {
    currentNode = *i;
    basis.col(j) = currentNode->BasisVector();
  }
}

template<typename MatType>
inline void CosineTree<MatType>::CosineNodeSplit()
{
  // If less than two points, splitting does not make sense---there is nothing
  // to split.
  if (numColumns < 2)
    return;

  // Calculate cosines with respect to the splitting point.
  VecType cosines;
  CalculateCosines(cosines);

  // Compute maximum and minimum cosine values.
  double cosineMax, cosineMin;
  cosineMax = arma::max(cosines % (cosines < 1));
  cosineMin = min(cosines);

  std::vector<size_t> leftIndices, rightIndices;

  // Split columns into left and right children. The splitting condition for the
  // column to be in the left child is as follows:
  //       cos_max - cos(i) < cos(i) - cos_min
  // We deviate from the paper here and use < instead of <= in order to handle
  // the edge case where cosineMax == cosineMin, and force there to be at least
  // one point in the right node.
  for (size_t i = 0; i < numColumns; ++i)
  {
    if (cosineMax - cosines(i) < cosines(i) - cosineMin)
      leftIndices.push_back(i);
    else
      rightIndices.push_back(i);
  }

  // Split the node into left and right children.
  left = new CosineTree(*this, leftIndices);
  right = new CosineTree(*this, rightIndices);
}

template<typename MatType>
inline void CosineTree<MatType>::ColumnSamplesLS(
    std::vector<size_t>& sampledIndices,
    typename CosineTree<MatType>::VecType& probabilities,
    size_t numSamples)
{
  // Initialize the cumulative distribution vector size.
  VecType cDistribution;
  cDistribution.zeros(numColumns + 1);

  // Calculate cumulative length-squared distribution for the node.
  for (size_t i = 0; i < numColumns; ++i)
  {
    cDistribution(i + 1) = cDistribution(i) +
        (l2NormsSquared(i) / frobNormSquared);
  }

  // Initialize sizes of the 'sampledIndices' and 'probabilities' vectors.
  sampledIndices.resize(numSamples);
  probabilities.zeros(numSamples);

  for (size_t i = 0; i < numSamples; ++i)
  {
    // Generate a random value for sampling.
    double randValue = arma::randu();
    size_t start = 0, end = numColumns, searchIndex;

    // Sample from the distribution and store corresponding probability.
    searchIndex = BinarySearch(cDistribution, randValue, start, end);
    sampledIndices[i] = indices[searchIndex];
    probabilities(i) = l2NormsSquared(searchIndex) / frobNormSquared;
  }
}

template<typename MatType>
inline size_t CosineTree<MatType>::ColumnSampleLS()
{
  // If only one element is present, there can only be one sample.
  if (numColumns < 2)
  {
    return 0;
  }

  // Initialize the cumulative distribution vector size.
  VecType cDistribution;
  cDistribution.zeros(numColumns + 1);

  // Calculate cumulative length-squared distribution for the node.
  for (size_t i = 0; i < numColumns; ++i)
  {
    cDistribution(i + 1) = cDistribution(i) +
        (l2NormsSquared(i) / frobNormSquared);
  }

  // Generate a random value for sampling.
  double randValue = arma::randu();
  size_t start = 0, end = numColumns;

  // Sample from the distribution.
  return BinarySearch(cDistribution, randValue, start, end);
}

template<typename MatType>
inline size_t CosineTree<MatType>::BinarySearch(
    typename CosineTree<MatType>::VecType& cDistribution,
    double value,
    size_t start,
    size_t end)
{
  size_t pivot = (start + end) / 2;

  // If pivot is zero, first point is the sampled point.
  if (!pivot)
  {
    return pivot;
  }

  // Binary search recursive algorithm.
  if (value > cDistribution(pivot - 1) && value <= cDistribution(pivot))
  {
    return (pivot - 1);
  }
  else if (value < cDistribution(pivot - 1))
  {
    return BinarySearch(cDistribution, value, start, pivot - 1);
  }
  else
  {
    return BinarySearch(cDistribution, value, pivot + 1, end);
  }
}

template<typename MatType>
inline void CosineTree<MatType>::CalculateCosines(
    typename CosineTree<MatType>::VecType& cosines)
{
  // Initialize cosine vector as a vector of zeros.
  cosines.zeros(numColumns);

  for (size_t i = 0; i < numColumns; ++i)
  {
    // If norm is zero, store cosine value as zero. Else, calculate cosine value
    // between two vectors.
    if (l2NormsSquared(i) == 0)
    {
      cosines(i) = 0;
    }
    else
    {
      cosines(i) =
          std::abs(arma::norm_dot(dataset->col(indices[splitPointIndex]),
                                  dataset->col(indices[i])));
    }
  }
}

template<typename MatType>
inline void CosineTree<MatType>::CalculateCentroid()
{
  // Initialize centroid as vector of zeros.
  centroid.zeros(dataset->n_rows);

  // Calculate centroid of columns in the node.
  for (size_t i = 0; i < numColumns; ++i)
  {
    centroid += dataset->col(indices[i]);
  }
  centroid /= numColumns;
}

} // namespace mlpack

#endif
