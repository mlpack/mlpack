/**
 * @file cosine_tree_impl.hpp
 * @author Mudit Raj Gupta
 *
 * Implementation of cosine tree.
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
