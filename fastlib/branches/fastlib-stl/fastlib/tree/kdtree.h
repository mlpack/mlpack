/**
 * @file tree/kdtree.h
 *
 * Tools for kd-trees.
 *
 * Eventually we hope to support KD trees with non-L2 (Euclidean)
 * metrics, like Manhattan distance.
 *
 * @experimental
 */

#ifndef TREE_KDTREE_H
#define TREE_KDTREE_H

#include "../base/base.h"

#include "spacetree.h"
#include "bounds.h"

#include "../fx/io.h"

#include <armadillo>

/**
 * Regular pointer-style trees (as opposed to THOR trees).
 */
namespace mlpack {
namespace tree {
  /**
   * Creates a KD tree from data, splitting on the midpoint.
   *
   * @experimental
   *
   * This requires you to pass in two vectors which will contain index mappings
   * so you can account for the re-ordering of the matrix.
   *
   * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
   * @param split_dimensions ordering of the dimensions that we should split on;
   *        the first element in this vector will be the first element we split
   *        on, and so on
   * @param leaf_size the maximum points in a leaf
   * @param old_from_new vector that will map new indices to original
   * @param new_from_old vector that will map original indexes to new indices
   */
  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpointSelective(arma::Mat<T>& matrix, 
				       const arma::uvec& split_dimensions,
                                       index_t leaf_size,
                                       arma::Col<index_t>& old_from_new,
                                       arma::Col<index_t>& new_from_old);
  
  /**
   * Creates a KD tree from data, splitting on the midpoint.
   *
   * @experimental
   *
   * This requires you to pass in one vector that will map new indices to the
   * original.  It does not bother keeping track of a map of the original
   * indices to the new indices.
   *
   * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
   * @param split_dimensions ordering of the dimensions that we should split on;
   *        the first element in this vector will be the first element we split
   *        on, and so on
   * @param leaf_size the maximum points in a leaf
   * @param old_from_new vector that will map new indices to original
   */
  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpointSelective(arma::Mat<T>& matrix, 
				       const arma::uvec& split_dimensions,
                                       index_t leaf_size,
                                       arma::Col<index_t>& old_from_new);

  /**
   * Creates a KD tree from data, splitting on the midpoint.
   *
   * @experimental
   *
   * This does not keep track of a map of the old indices to the new indices.
   *
   * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
   * @param split_dimensions ordering of the dimensions that we should split on;
   *        the first element in this vector will be the first element we split
   *        on, and so on
   * @param leaf_size the maximum points in a leaf
   */
  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpointSelective(arma::Mat<T>& matrix, 
				       const arma::uvec& split_dimensions,
                                       index_t leaf_size);

  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix, 
                              index_t leaf_size,
                              arma::Col<index_t>& old_from_new,
                              arma::Col<index_t>& new_from_old);
  
  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix, 
                              index_t leaf_size,
                              arma::Col<index_t>& old_from_new);
  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix, index_t leaf_size);

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "kdtree_impl.h"

#endif
