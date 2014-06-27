/**
 * @file cosine_node_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of cosine node.
 */
#ifndef __MLPACK_CORE_TREE_COSINE_TREE_COSINE_NODE_IMPL_HPP
#define __MLPACK_CORE_TREE_COSINE_TREE_COSINE_NODE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "cosine_node.hpp"

namespace mlpack {
namespace tree {

CosineNode::CosineNode(const arma::mat& dataset) :
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

CosineNode::CosineNode(CosineNode& parentNode,
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

void CosineNode::CosineNodeSplit()
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
  // 			cos_max - cos(i) <= cos(i) - cos_min
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
  left = new CosineNode(*this, leftIndices);
  right = new CosineNode(*this, rightIndices);
}

void CosineNode::ColumnSamplesLS(std::vector<size_t>& sampledIndices,
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

size_t CosineNode::ColumnSampleLS()
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

size_t CosineNode::BinarySearch(arma::vec& cDistribution,
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

void CosineNode::CalculateCosines(arma::vec& cosines)
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

void CosineNode::CalculateCentroid()
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

#endif
