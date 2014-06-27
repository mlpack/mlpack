/**
 * @file cosine_tree.hpp
 * @author Siddharth Agrawal
 *
 * Definition of Cosine Tree.
 */
 
#ifndef __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_HPP
#define __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_HPP

#include <mlpack/core.hpp>
#include <boost/heap/priority_queue.hpp>

#include "cosine_node.hpp"

namespace mlpack {
namespace tree {

class CosineTree
{
 public:
 
  // Type definition for CosineNode priority queue.
  typedef boost::heap::priority_queue<CosineNode*,
      boost::heap::compare<CompareCosineNode> > CosineNodeQueue;
 
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
  double MonteCarloError(CosineNode* node,
                         CosineNodeQueue& treeQueue,
                         arma::vec* addBasisVector1 = NULL,
                         arma::vec* addBasisVector2 = NULL);
  
  /**
   * Constructs the final basis matrix, after the cosine tree construction.
   *
   * @param treeQueue Priority queue of cosine nodes.
   */                       
  void ConstructBasis(CosineNodeQueue& treeQueue);
  
  //! Returns the basis of the constructed subspace.
  void GetFinalBasis(arma::mat& finalBasis) { finalBasis = basis; }
  
 private:
  //! Matrix for which cosine tree is constructed.
  const arma::mat& dataset;
  //! Error tolerance fraction for calculated subspace.
  double epsilon;
  //! Cumulative probability for Monte Carlo error lower bound.
  double delta;
  //! Subspace basis of the input dataset.
  arma::mat basis;
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "cosine_tree_impl.hpp"

#endif
