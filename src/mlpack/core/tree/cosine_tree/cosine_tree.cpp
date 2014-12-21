/**
 * @file cosine_tree_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of cosine tree.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "cosine_tree.hpp"

#include <boost/math/distributions/normal.hpp>

namespace mlpack {
namespace tree {

CosineTree::CosineTree(const arma::mat& dataset) :
    dataset(dataset),
    parent(NULL),
    right(NULL),
    left(NULL),
    numColumns(dataset.n_cols)
{  
  // Initialize sizes of column indices and l2 norms.
  indices.resize(numColumns);
  l2NormsSquared.zeros(numColumns);
  
  // Set indices and calculate squared norms of the columns.
  for(size_t i = 0; i < numColumns; i++)
  {
    indices[i] = i;
    double l2Norm = arma::norm(dataset.col(i), 2);
    l2NormsSquared(i) = l2Norm * l2Norm;
  }
  
  // Frobenius norm of columns in the node.
  frobNormSquared = arma::accu(l2NormsSquared);
  
  // Calculate centroid of columns in the node.
  CalculateCentroid();
  
  splitPointIndex = ColumnSampleLS();
}

CosineTree::CosineTree(CosineTree& parentNode,
                       const std::vector<size_t>& subIndices) :
    dataset(parentNode.GetDataset()),
    parent(&parentNode),
    right(NULL),
    left(NULL),
    numColumns(subIndices.size())
{
  // Initialize sizes of column indices and l2 norms.
  indices.resize(numColumns);
  l2NormsSquared.zeros(numColumns);
  
  // Set indices and squared norms of the columns.
  for(size_t i = 0; i < numColumns; i++)
  {
    indices[i] = parentNode.indices[subIndices[i]];
    l2NormsSquared(i) = parentNode.l2NormsSquared(subIndices[i]);
  }
  
  // Frobenius norm of columns in the node.
  frobNormSquared = arma::accu(l2NormsSquared);
  
  // Calculate centroid of columns in the node.
  CalculateCentroid();
  
  splitPointIndex = ColumnSampleLS();
}

CosineTree::CosineTree(const arma::mat& dataset,
                       const double epsilon,
                       const double delta) :
    dataset(dataset),
    epsilon(epsilon),
    delta(delta)
{
  // Declare the cosine tree priority queue.
  CosineNodeQueue treeQueue;
  
  // Define root node of the tree and add it to the queue.
  CosineTree root(dataset);
  arma::vec tempVector = arma::zeros(dataset.n_rows);
  root.L2Error(0);
  root.BasisVector(tempVector);
  treeQueue.push(&root);
  
  // Initialize Monte Carlo error estimate for comparison.
  double monteCarloError = root.FrobNormSquared();
  
  while(monteCarloError > epsilon * root.FrobNormSquared())
  {
    // Pop node from queue with highest projection error.
    CosineTree* currentNode;
    currentNode = treeQueue.top();
    treeQueue.pop();
    
    // Split the node into left and right children.
    currentNode->CosineNodeSplit();
    
    // Obtain pointers to the left and right children of the current node.
    CosineTree *currentLeft, *currentRight;
    currentLeft = currentNode->Left();
    currentRight = currentNode->Right();
    
    // Calculate basis vectors of left and right children.
    arma::vec lBasisVector, rBasisVector;
    
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
    treeQueue.push(currentLeft);
    treeQueue.push(currentRight);
    
    // Calculate Monte Carlo error estimate for the root node.
    monteCarloError = MonteCarloError(&root, treeQueue);
  }
  
  // Construct the subspace basis from the current priority queue.
  ConstructBasis(treeQueue);
}

void CosineTree::ModifiedGramSchmidt(CosineNodeQueue& treeQueue,
                                     arma::vec& centroid,
                                     arma::vec& newBasisVector,
                                     arma::vec* addBasisVector)
{
  // Set new basis vector to centroid.
  newBasisVector = centroid;

  // Variables for iterating throught the priority queue.
  CosineTree *currentNode;
  CosineNodeQueue::const_iterator i = treeQueue.begin();

  // For every vector in the current basis, remove its projection from the
  // centroid.
  for(; i != treeQueue.end(); i++)
  {
    currentNode = *i;
    
    double projection = arma::dot(currentNode->BasisVector(), centroid);
    newBasisVector -= projection * currentNode->BasisVector();
  }
  
  // If additional basis vector is passed, take it into account.
  if(addBasisVector)
  {
    double projection = arma::dot(*addBasisVector, centroid);
    newBasisVector -= *addBasisVector * projection;
  }
  
  // Normalize the modified centroid vector.
  if(arma::norm(newBasisVector, 2))
    newBasisVector /= arma::norm(newBasisVector, 2);
}

double CosineTree::MonteCarloError(CosineTree* node,
                                   CosineNodeQueue& treeQueue,
                                   arma::vec* addBasisVector1,
                                   arma::vec* addBasisVector2)
{
  std::vector<size_t> sampledIndices;
  arma::vec probabilities;
  
  // Sample O(log m) points from the input node's distribution.
  // 'm' is the number of columns present in the node.
  size_t numSamples = log(node->NumColumns()) + 1;  
  node->ColumnSamplesLS(sampledIndices, probabilities, numSamples);
  
  // Get pointer to the original dataset.
  arma::mat dataset = node->GetDataset();
  
  // Initialize weighted projection magnitudes as zeros.
  arma::vec weightedMagnitudes;
  weightedMagnitudes.zeros(numSamples);
  
  // Set size of projection vector, depending on whether additional basis
  // vectors are passed.
  size_t projectionSize;
  if(addBasisVector1 && addBasisVector2)
    projectionSize = treeQueue.size() + 2;
  else
    projectionSize = treeQueue.size();
  
  // For each sample, calculate the weighted projection onto the current basis.
  for(size_t i = 0; i < numSamples; i++)
  {
    // Initialize projection as a vector of zeros.
    arma::vec projection;
    projection.zeros(projectionSize);

    CosineTree *currentNode;
    CosineNodeQueue::const_iterator j = treeQueue.begin();
  
    size_t k = 0;
    // Compute the projection of the sampled vector onto the existing subspace.
    for(; j != treeQueue.end(); j++, k++)
    {
      currentNode = *j;
    
      projection(k) = arma::dot(dataset.col(sampledIndices[i]),
                                currentNode->BasisVector());
    }
    // If two additional vectors are passed, take their projections.
    if(addBasisVector1 && addBasisVector2)
    {
      projection(k++) = arma::dot(dataset.col(sampledIndices[i]),
                                  *addBasisVector1);
      projection(k) = arma::dot(dataset.col(sampledIndices[i]),
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
  
  if(!sigma)
  {
    node->L2Error(node->FrobNormSquared() - mu);
    return (node->FrobNormSquared() - mu);
  }
  
  // Fit a normal distribution using the calculated statistics, and calculate a
  // lower bound on the magnitudes for the passed 'delta' parameter.
  boost::math::normal dist(mu, sigma);
  double lowerBound = boost::math::quantile(dist, delta);
  
  // Upper bound on the subspace reconstruction error.
  node->L2Error(node->FrobNormSquared() - lowerBound);
  
  return (node->FrobNormSquared() - lowerBound);
}

void CosineTree::ConstructBasis(CosineNodeQueue& treeQueue)
{
  // Initialize basis as matrix of zeros.
  basis.zeros(dataset.n_rows, treeQueue.size());
  
  // Variables for iterating through the priority queue.
  CosineTree *currentNode;
  CosineNodeQueue::const_iterator i = treeQueue.begin();
  
  // Transfer basis vectors from the queue to the basis matrix.
  size_t j = 0;
  for(; i != treeQueue.end(); i++, j++)
  {
    currentNode = *i;
    basis.col(j) = currentNode->BasisVector();
  }
}

void CosineTree::CosineNodeSplit()
{
  //! If less than two nodes, splitting does not make sense.
  if(numColumns < 3) return;
  
  //! Calculate cosines with respect to the splitting point.
  arma::vec cosines;
  CalculateCosines(cosines);
  
  //! Compute maximum and minimum cosine values.
  double cosineMax, cosineMin;
  cosineMax = arma::max(cosines % (cosines < 1));
  cosineMin = arma::min(cosines);
  
  std::vector<size_t> leftIndices, rightIndices;
  
  // Split columns into left and right children. The splitting condition for the
  // column to be in the left child is as follows:
  //       cos_max - cos(i) <= cos(i) - cos_min
  for(size_t i = 0; i < numColumns; i++)
  {
    if(cosineMax - cosines(i) <= cosines(i) - cosineMin)
    {
      leftIndices.push_back(i);
    }
    else
    {
      rightIndices.push_back(i);
    }
  }
  
  // Split the node into left and right children.
  left = new CosineTree(*this, leftIndices);
  right = new CosineTree(*this, rightIndices);
}

void CosineTree::ColumnSamplesLS(std::vector<size_t>& sampledIndices,
                                 arma::vec& probabilities,
                                 size_t numSamples)
{
  // Initialize the cumulative distribution vector size.
  arma::vec cDistribution;
  cDistribution.zeros(numColumns + 1);
  
  // Calculate cumulative length-squared distribution for the node.
  for(size_t i = 0; i < numColumns; i++)
  {
    cDistribution(i+1) = cDistribution(i) + l2NormsSquared(i) / frobNormSquared;
  }
  
  // Intialize sizes of the 'sampledIndices' and 'probabilities' vectors.
  sampledIndices.resize(numSamples);
  probabilities.zeros(numSamples);
  
  for(size_t i = 0; i < numSamples; i++)
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

size_t CosineTree::ColumnSampleLS()
{
  // If only one element is present, there can only be one sample.
  if(numColumns < 2)
  {
    return 0;
  }

  // Initialize the cumulative distribution vector size.
  arma::vec cDistribution;
  cDistribution.zeros(numColumns + 1);
  
  // Calculate cumulative length-squared distribution for the node.
  for(size_t i = 0; i < numColumns; i++)
  {
    cDistribution(i+1) = cDistribution(i) + l2NormsSquared(i) / frobNormSquared;
  }
  
  // Generate a random value for sampling.
  double randValue = arma::randu();
  size_t start = 0, end = numColumns;
  
  // Sample from the distribution.
  return BinarySearch(cDistribution, randValue, start, end);
}

size_t CosineTree::BinarySearch(arma::vec& cDistribution,
                                double value,
                                size_t start,
                                size_t end)
{
  size_t pivot = (start + end) / 2;
  
  // If pivot is zero, first point is the sampled point.
  if(!pivot)
  {
    return pivot;
  }
  
  // Binary search recursive algorithm.
  if(value > cDistribution(pivot - 1) && value <= cDistribution(pivot))
  {
    return (pivot - 1);
  }
  else if(value < cDistribution(pivot - 1))
  {
    return BinarySearch(cDistribution, value, start, pivot - 1);
  }
  else
  {
    return BinarySearch(cDistribution, value, pivot + 1, end);
  }
}

void CosineTree::CalculateCosines(arma::vec& cosines)
{
  // Initialize cosine vector as a vector of zeros.
  cosines.zeros(numColumns);
  
  for(size_t i = 0; i < numColumns; i++)
  {
    // If norm is zero, store cosine value as zero. Else, calculate cosine value
    // between two vectors.
    if(l2NormsSquared(i) == 0)
    {
      cosines(i) = 0;
    }
    else
    {
      cosines(i) = arma::norm_dot(dataset.col(indices[splitPointIndex]),
                                  dataset.col(indices[i]));
    }
  }
}

void CosineTree::CalculateCentroid()
{
  // Initialize centroid as vector of zeros.
  centroid.zeros(dataset.n_rows);
  
  // Calculate centroid of columns in the node.
  for(size_t i = 0; i < numColumns; i++)
  {
    centroid += dataset.col(indices[i]);
  }
  centroid /= numColumns;
}

}; // namespace tree
}; // namespace mlpack
