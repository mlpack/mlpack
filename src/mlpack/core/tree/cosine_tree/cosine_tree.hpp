/**
 * @file cosine_tree.hpp
 * @author Mudit Raj Gupta
 *
 * Definition of Cosine Tree.
 *
 * This file is part of MLPACK 1.0.7.
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

namespace mlpack {
namespace tree /** Cosine Trees and building procedures. */ {

class CosineTree
{
 private:
  //! Data. 
  arma::mat data;
  //! Centroid.
  arma::rowvec centroid;
  //! Sampling Probabilities
  arma::vec probabilities;
  //! The left child node.
  CosineTree* left;
  //! The right child node.
  CosineTree* right;
  //! Number of points in the node.
  size_t numPoints;
  
 public:
  //! So other classes can use TreeType::Mat.
  //typedef MatType Mat;
  /**
   * Constructor 
   * 
   * @param data Dataset to create tree from. 
   * @param centroid Centroid of the matrix.
   * @param probabilities Sampling probabilities
   */
  CosineTree(arma::mat data, arma::rowvec centroid, arma::vec probabilities);

  /**
   * Create an empty tree node.
   */
  CosineTree();

  /**
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any pointers or references
   * to any nodes which are children of this one.
   */
  ~CosineTree();

  //! Gets the left child of this node.
  CosineTree* Left() const;
  
  //!Sets the Left child of this node.
  void Left(CosineTree* child);

  //! Gets the right child of this node.
  CosineTree* Right() const;

  //!Sets the Right child of this node.
  void Right(CosineTree* child);

  /**
   * Return the specified child (0 will be left, 1 will be right).  If the index
   * is greater than 1, this will return the right child.
   *
   * @param child Index of child to return.
   */
  CosineTree& Child(const size_t child) const;

  //! Return the number of points in this node (0 if not a leaf).
  size_t NumPoints() const;

  //! Returns a reference to the data
  arma::mat Data();
  
  //! Sets a reference to the data
  void Data(arma::mat& d);
  
  //! Returns a reference to Sample Probabilites
  arma::vec Probabilities();
  
  //! Sets a reference to Sample Probabilites
  void Probabilities(arma::vec& prob);
  
  //! Returns a reference to the centroid
  arma::rowvec Centroid();  
  
  //! Sets the centroid
  void Centroid(arma::rowvec& centr);  
   
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "cosine_tree_impl.hpp"

#endif
