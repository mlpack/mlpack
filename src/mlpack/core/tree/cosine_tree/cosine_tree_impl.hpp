/**
 * @file cosine_tree_impl.hpp
 * @author Mudit Raj Gupta
 *
 * Implementation of cosine tree.
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
#ifndef __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP

#include "cosine_tree.hpp"

namespace mlpack {
namespace tree {

CosineTree::CosineTree(arma::mat data, arma::rowvec centroid, arma::vec probabilities) : 
    data(data.t()),
    centroid(centroid),
    probabilities(probabilities),
    left(NULL),
    right(NULL),
    numPoints(data.n_cols)
{ 
  // Nothing to do
}

CosineTree::CosineTree() : 
    left(NULL),
    right(NULL)
{   
  // Nothing to do
}

CosineTree::~CosineTree()
{
  if (left)
    delete left;
  if (right)
    delete right;
}

CosineTree* CosineTree::Right() const
{
  return right;
}

void CosineTree::Right(CosineTree* child)
{
  right = child;
}

CosineTree* CosineTree::Left() const
{
  return left;
}

void CosineTree::Left(CosineTree* child)
{
  left = child;
}

CosineTree& CosineTree::Child(const size_t child) const
{
  if (child == 0)
    return *left;
  else
    return *right;
}

size_t CosineTree::NumPoints() const
{
  return numPoints;
}

arma::mat CosineTree::Data()
{
  return data;
}

void CosineTree::Data(arma::mat& d)
{
    data = d;
    numPoints = d.n_rows;
}

arma::vec CosineTree::Probabilities() 
{
  return probabilities;
}

void CosineTree::Probabilities(arma::vec& prob)
{
  probabilities = prob; 
}

arma::rowvec CosineTree::Centroid()
{
  return centroid;
}

void CosineTree::Centroid(arma::rowvec& centr)
{
    centroid = centr;
}

}; // namespace tree
}; // namespace mlpack

#endif
