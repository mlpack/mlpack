/**
 * @file cosine_tree.hpp
 * @author Siddharth Agrawal
 *
 * Definition of Cosine Tree.
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
 
#ifndef __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_HPP
#define __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_HPP

#include <mlpack/core.hpp>
#include <boost/heap/priority_queue.hpp>

namespace mlpack {
namespace tree {

// Predeclare classes for CosineNodeQueue typedef.
class CompareCosineNode;
class CosineTree;

// CosineNodeQueue typedef.
typedef boost::heap::priority_queue<CosineTree*,
    boost::heap::compare<CompareCosineNode> > CosineNodeQueue;

class CosineTree
{
 public:
      
  /**
   * CosineTree constructor for the root node of the tree. It initializes the
   * necessary variables required for splitting of the node, and building the
   * tree further. It takes a pointer to the input matrix and calculates the
   * relevant variables using it.
   *
   * @param dataset Matrix for which cosine tree is constructed.
   */
  CosineTree(const arma::mat& dataset);
  
  /**
   * CosineTree constructor for nodes other than the root node of the tree. It
   * takes in a pointer to the parent node and a list of column indices which
   * mentions the columns to be included in the node. The function calculate the
   * relevant variables just like the constructor above.
   *
   * @param parentNode Pointer to the parent cosine node.
   * @param subIndices Pointer to vector of column indices to be included.
   */
  CosineTree(CosineTree& parentNode, const std::vector<size_t>& subIndices);
 
  /**
   * Construct the CosineTree and the basis for the given matrix, and passed
   * 'epsilon' and 'delta' parameters. The CosineTree is constructed by
   * splitting nodes in the direction of maximum error, stored using a priority
   * queue. Basis vectors are added from the left and right children of the
   * split node. The basis vector from a node is the orthonormalized centroid of
   * its columns. The splitting continues till the Monte Carlo estimate of the
   * input matrix's projection on the obtained subspace is less than a fraction
   * of the norm of the input matrix.
   *
   * @param dataset Matrix for which the CosineTree is constructed.
   * @param epsilon Error tolerance fraction for calculated subspace.
   * @param delta Cumulative probability for Monte Carlo error lower bound.
   */
  CosineTree(const arma::mat& dataset,
             const double epsilon,
             const double delta);
  
  /**
   * Calculates the orthonormalization of the passed centroid, with respect to
   * the current vector subspace.
   *
   * @param treeQueue Priority queue of cosine nodes.
   * @param centroid Centroid of the node being added to the basis.
   * @param newBasisVector Orthonormalized centroid of the node.
   * @param addBasisVector Address to additional basis vector.
   */                           
  void ModifiedGramSchmidt(CosineNodeQueue& treeQueue,
                           arma::vec& centroid,
                           arma::vec& newBasisVector,
                           arma::vec* addBasisVector = NULL);
  
  /**
   * Estimates the squared error of the projection of the input node's matrix
   * onto the current vector subspace. A normal distribution is fit using
   * weighted norms of projections of samples drawn from the input node's matrix
   * columns. The error is calculated as the difference between the Frobenius
   * norm of the input node's matrix and lower bound of the normal distribution.
   *
   * @param node Node for which Monte Carlo estimate is calculated.
   * @param treeQueue Priority queue of cosine nodes.
   * @param addBasisVector1 Address to first additional basis vector.
   * @param addBasisVector2 Address to second additional basis vector.
   */                         
  double MonteCarloError(CosineTree* node,
                         CosineNodeQueue& treeQueue,
                         arma::vec* addBasisVector1 = NULL,
                         arma::vec* addBasisVector2 = NULL);
  
  /**
   * Constructs the final basis matrix, after the cosine tree construction.
   *
   * @param treeQueue Priority queue of cosine nodes.
   */                       
  void ConstructBasis(CosineNodeQueue& treeQueue);
                         
  /**
   * This function splits the cosine node into two children based on the cosines
   * of the columns contained in the node, with respect to the sampled splitting
   * point. The function also calls the CosineTree constructor for the children.
   */
  void CosineNodeSplit();
                         
  /**
   * Sample 'numSamples' points from the Length-Squared distribution of the
   * cosine node. The function uses 'l2NormsSquared' to calculate the cumulative
   * probability distribution of the column vectors. The sampling is based on a
   * randomly generated values in the range [0, 1].
   */
  void ColumnSamplesLS(std::vector<size_t>& sampledIndices, 
                       arma::vec& probabilities, size_t numSamples);
  
  /**
   * Sample a point from the Length-Squared distribution of the cosine node. The
   * function uses 'l2NormsSquared' to calculate the cumulative probability
   * distribution of the column vectors. The sampling is based on a randomly
   * generated value in the range [0, 1].
   */
  size_t ColumnSampleLS();
  
  /**
   * Sample a column based on the cumulative Length-Squared distribution of the
   * cosine node, and a randomly generated value in the range [0, 1]. Binary
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
  
  //! Returns the basis of the constructed subspace.
  void GetFinalBasis(arma::mat& finalBasis) { finalBasis = basis; }
  
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
  CosineTree* Left() { return left; }
  
  //! Get pointer to the right child of the node.
  CosineTree* Right() { return right; }
  
  //! Get number of columns of input matrix in the node.
  size_t NumColumns() const { return numColumns; }
  
  //! Get the Frobenius norm squared of columns in the node.
  double FrobNormSquared() const { return frobNormSquared; }
  
  //! Get the column index of split point of the node.
  size_t SplitPointIndex() const { return indices[splitPointIndex]; }
  
 private:
  //! Matrix for which cosine tree is constructed.
  const arma::mat& dataset;
  //! Error tolerance fraction for calculated subspace.
  double epsilon;
  //! Cumulative probability for Monte Carlo error lower bound.
  double delta;
  //! Subspace basis of the input dataset.
  arma::mat basis;
  //! Parent of the node.
  CosineTree* parent;
  //! Right child of the node.
  CosineTree* right;
  //! Left child of the node.
  CosineTree* left;
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
};

class CompareCosineNode
{
 public:
 
  // Comparison function for construction of priority queue.
  bool operator() (const CosineTree* a, const CosineTree* b) const
  {
    return a->L2Error() < b->L2Error();
  }
};

}; // namespace tree
}; // namespace mlpack

#endif
