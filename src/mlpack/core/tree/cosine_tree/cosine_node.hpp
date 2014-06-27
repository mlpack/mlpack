/**
 * @file cosine_node.hpp
 * @author Siddharth Agrawal
 *
 * Definition of Cosine Node.
 */
 
#ifndef __MLPACK_CORE_TREE_COSINE_TREE_COSINE_NODE_HPP
#define __MLPACK_CORE_TREE_COSINE_TREE_COSINE_NODE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

class CosineNode
{
 public:
 
  /**
   * CosineNode constructor for the root node of the tree. It initializes the
   * necessary variables required for splitting of the node, and building the
   * tree further. It takes a pointer to the input matrix and calculates the
   * relevant variables using it.
   *
   * @param dataset Matrix for which cosine tree is constructed.
   */
  CosineNode(const arma::mat& dataset);
  
  /**
   * CosineNode constructor for nodes other than the root node of the tree. It
   * takes in a pointer to the parent node and a list of column indices which
   * mentions the columns to be included in the node. The function calculate the
   * relevant variables just like the constructor above.
   *
   * @param parentNode Pointer to the parent CosineNode.
   * @param subIndices Pointer to vector of column indices to be included.
   */
  CosineNode(CosineNode& parentNode, const std::vector<size_t>& subIndices);
  
  /**
   * This function splits the CosineNode into two children based on the cosines
   * of the columns contained in the node, with respect to the sampled splitting
   * point. The function also calls the CosineNode constructor for the children.
   */
  void CosineNodeSplit();
  
  /**
   * Sample 'numSamples' points from the Length-Squared distribution of the
   * CosineNode. The function uses 'l2NormsSquared' to calculate the cumulative
   * probability distribution of the column vectors. The sampling is based on a
   * randomly generated values in the range [0, 1].
   */
  void ColumnSamplesLS(std::vector<size_t>& sampledIndices, 
                       arma::vec& probabilities, size_t numSamples);
  
  /**
   * Sample a point from the Length-Squared distribution of the CosineNode. The
   * function uses 'l2NormsSquared' to calculate the cumulative probability
   * distribution of the column vectors. The sampling is based on a randomly
   * generated value in the range [0, 1].
   */
  size_t ColumnSampleLS();
  
  /**
   * Sample a column based on the cumulative Length-Squared distribution of the
   * CosineNode, and a randomly generated value in the range [0, 1]. Binary
   * search is more efficient than searching linearly for the same. This leads
   * a significant speedup when there are large number of columns to choose from
   * and when a number of samples are to be drawn from the distribution.
   *
   * @param cDistribution Cumulative LS distibution of columns in the node.
   * @param value Randomly generated value in the range [0, 1].
   * @param start Starting index of the distribution interval to search in.
   * @param end Ending index of the distribution interval to search in.
   */
  size_t BinarySearch(arma::vec& cDistribution, double value, size_t start,
                      size_t end);
  
  /**
   * Calculate cosines of the columns present in the node, with respect to the
   * sampled splitting point. The calculated cosine values are useful for
   * splitting the node into its children.
   *
   * @param cosines Vector to store the cosine values in.
   */
  void CalculateCosines(arma::vec& cosines);
  
  /**
   * Calculate centroid of the columns present in the node. The calculated
   * centroid is used as a basis vector for the cosine tree being constructed.
   */
  void CalculateCentroid();
  
  //! Get pointer to the dataset matrix.
  const arma::mat& GetDataset() const { return dataset; }
  
  //! Get the indices of columns in the node.
  std::vector<size_t>& VectorIndices() { return indices; }
  
  //! Set the Monte Carlo error.
  void L2Error(const double error) { this->l2Error = error; }
  
  //! Get the Monte Carlo error.
  double L2Error() const { return l2Error; }
  
  //! Get pointer to the centroid vector.
  arma::vec& Centroid() { return centroid; }
  
  //! Set the basis vector of the node.
  void BasisVector(arma::vec& bVector) { this->basisVector = bVector; }
  
  //! Get the basis vector of the node.
  arma::vec& BasisVector() { return basisVector; }
  
  //! Get pointer to the left child of the node.
  CosineNode* Left() { return left; }
  
  //! Get pointer to the right child of the node.
  CosineNode* Right() { return right; }
  
  //! Get number of columns of input matrix in the node.
  size_t NumColumns() const { return numColumns; }
  
  //! Get the Frobenius norm squared of columns in the node.
  double FrobNormSquared() const { return frobNormSquared; }
  
  //! Get the column index of split point of the node.
  size_t SplitPointIndex() const { return indices[splitPointIndex]; }
 
 private:
  //! Matrix for which cosine tree is constructed.
  const arma::mat& dataset;
  //! Parent of the node.
  CosineNode* parent;
  //! Right child of the node.
  CosineNode* right;
  //! Left child of the node.
  CosineNode* left;
  //! Indices of columns of input matrix in the node.
  std::vector<size_t> indices;
  //! L2-norm squared of columns in the node.
  arma::vec l2NormsSquared;
  //! Centroid of columns of input matrix in the node.
  arma::vec centroid;
  //! Orthonormalized basis vector of the node.
  arma::vec basisVector;
  //! Index of split point of cosine node.
  size_t splitPointIndex;
  //! Number of columns of input matrix in the node.
  size_t numColumns;
  //! Monte Carlo error for this node.
  double l2Error;
  //! Frobenius norm squared of columns in the node.
  double frobNormSquared;
  
  // Friend class to facilitate construction of priority queue.
  friend class CompareCosineNode;
};

class CompareCosineNode
{
 public:
 
  // Comparison function for construction of priority queue.
  bool operator() (const CosineNode* a, const CosineNode* b) const
  {
    return a->l2Error < b->l2Error;
  }
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "cosine_node_impl.hpp"

#endif
