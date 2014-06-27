/**
 * @file cosine_tree_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of cosine tree.
 */
#ifndef __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "cosine_tree.hpp"

#include <boost/math/distributions/normal.hpp>

namespace mlpack {
namespace tree {

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
  CosineNode root(dataset);
  arma::vec tempVector = arma::zeros(dataset.n_rows);
  root.L2Error(0);
  root.BasisVector(tempVector);
  treeQueue.push(&root);
  
  // Initialize Monte Carlo error estimate for comparison.
  double monteCarloError = root.FrobNormSquared();
  
  while(monteCarloError > epsilon * root.FrobNormSquared())
  {
    // Pop node from queue with highest projection error.
    CosineNode* currentNode;
    currentNode = treeQueue.top();
    treeQueue.pop();
    
    // Split the node into left and right children.
    currentNode->CosineNodeSplit();
    
    // Obtain pointers to the left and right children of the current node.
    CosineNode *currentLeft, *currentRight;
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
    
    std::cout << monteCarloError / root.FrobNormSquared() << "\n";
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
  CosineNode *currentNode;
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

double CosineTree::MonteCarloError(CosineNode* node,
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

    CosineNode *currentNode;
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
  CosineNode *currentNode;
  CosineNodeQueue::const_iterator i = treeQueue.begin();
  
  // Transfer basis vectors from the queue to the basis matrix.
  size_t j = 0;
  for(; i != treeQueue.end(); i++, j++)
  {
    currentNode = *i;
    basis.col(j) = currentNode->BasisVector();
  }
}

}; // namespace tree
}; // namespace mlpack

#endif
