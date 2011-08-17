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

#include "kdtree_impl.h"

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
                                       arma::Col<index_t>& new_from_old) {
    TKdTree *node = new TKdTree();
   
    old_from_new.set_size(matrix.n_cols);  
    for (index_t i = 0; i < matrix.n_cols; i++)
      old_from_new[i] = i;
      
    node->Init(0, matrix.n_cols);
    node->bound().SetSize(split_dimensions.n_elem);
    tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
        0, matrix.n_cols, &node->bound());

    tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions, 
	node, leaf_size, old_from_new);
    
    if(new_from_old) {
      new_from_old.set_size(matrix.n_cols);
      for (index_t i = 0; i < matrix.n_cols; i++) {
        new_from_old[old_from_new[i]] = i;
      }
    }
    return node;
  }
  
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
                                       arma::Col<index_t>& old_from_new) {
    TKdTree *node = new TKdTree();
    
    old_from_new.set_size(matrix.n_cols);   
    for (index_t i = 0; i < matrix.n_cols; i++)
      old_from_new[i] = i;

    node->Init(0, matrix.n_cols);
    node->bound().SetSize(split_dimensions.n_elem);
    tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
        0, matrix.n_cols, &node->bound());

    tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions, 
	node, leaf_size, old_from_new);
    
    return node;
  }

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
                                       index_t leaf_size) {
    TKdTree *node = new TKdTree();

    node->Init(0, matrix.n_cols);
    node->bound().SetSize(split_dimensions.n_elem);
    tree_kdtree_private::SelectFindBoundFromMatrix(matrix, split_dimensions,
        0, matrix.n_cols, &node->bound());

    tree_kdtree_private::SelectSplitKdTreeMidpoint(matrix, split_dimensions, 
	node, leaf_size);
    
    return node;
  }

  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix, 
                              index_t leaf_size,
                              arma::Col<index_t>& old_from_new,
                              arma::Col<index_t>& new_from_old) {  
    // create vector of dimensions that we will split on
    // by default we'll just split the first dimension first, and so on
    arma::uvec split_dimensions(matrix.n_rows);
    for(int i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;

    TKdTree *result;
    result = MakeKdTreeMidpointSelective<TKdTree>(matrix, 
      split_dimensions, leaf_size, old_from_new, new_from_old);

    return result;
  }
  
  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix, 
                              index_t leaf_size,
                              arma::Col<index_t>& old_from_new) {  
    // create vector of dimensions that we will split on
    // by default we'll just split the first dimension first, and so on
    arma::uvec split_dimensions(matrix.n_rows);
    for(index_t i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;

    TKdTree *result;
    result = MakeKdTreeMidpointSelective<TKdTree>(matrix, 
      split_dimensions, leaf_size, old_from_new);

    return result;
  }
  
  template<typename TKdTree, typename T>
  TKdTree *MakeKdTreeMidpoint(arma::Mat<T>& matrix, 
                              index_t leaf_size) {  
    // create vector of dimensions that we will split on
    // by default we'll just split the first dimension first, and so on
    arma::uvec split_dimensions(matrix.n_rows);
    for(int i = 0; i < matrix.n_rows; i++)
      split_dimensions[i] = i;

    TKdTree *result;
    result = MakeKdTreeMidpointSelective<TKdTree>(matrix, 
      split_dimensions, leaf_size);

    return result;
  }

 /*** UNDOCUMENTED PARAMETERS ***/
 PARAM_STRING("q", "Specifies input file.", "kdtree", "foo.txt");
 PARAM_STRING("type", "Undocumented", "kdtree", "text");
 PARAM_INT("leaflen", "Leaf size.", "kdtree", 20);
 

  /**
   * Loads a KD tree from a command-line parameter,
   * creating a KD tree if necessary.
   *
   * @experimental
   *
   * This optionally allows the end user to write out the created KD tree
   * to a file, as a convenience.
   *
   * Requires a sub-module, with the root parameter of the submodule being
   * the filename, and optional parameters leaflen, type, and save (see
   * example below).
   *
   * Example:
   *
   * @code
   * MyKdTree *q_tree;
   * Matrix q_matrix;
   * arma::Col<index_t> q_permutation;
   * LoadKdTree(fx_submodule(NULL, "q", "q"), &q_matrix, &q_tree,
   *    &q_permutation);
   * @endcode
   *
   * Command-line use:
   *
   * @code
   * ./main --q=foo.txt                  # load from csv format
   * ./main --q=foo.txt --q/leaflen=20   # leaf length
   * @endcode
   *
   * @param module the module to get parameters from
   * @param matrix the matrix to initialize, undefined on failure
   * @param tree_pp an unitialized pointer that will be set to the root
   *        of the tree, must still be freed on failure
   * @param old_from_new stores the permutation to get from the indices in
   *        the matrix returned to the original data point indices
   * @return SUCCESS_PASS or SUCCESS_FAIL
   */
  template<typename TKdTree, typename T>
  success_t LoadKdTree(arma::Mat<T>& matrix,
                       TKdTree** tree_pp,
                       arma::Col<index_t>& old_from_new) {
    const char *type = mlpack::IO::GetParam<std::string>("kdtree/type").c_str();
    const char *fname = mlpack::IO::GetParam<std::string>("kdtree/q").c_str();
    success_t success = SUCCESS_PASS;

    mlpack::IO::StartTimer("kdtree/load");
    if (strcmp(type, "text") == 0) {
      int leaflen = mlpack::IO::GetParam<int>("kdtree/leaflen");

      mlpack::IO::StartTimer("kdtree/load_matrix");
      success = data::Load(fname, matrix);
      mlpack::IO::StopTimer("kdtree/load_matrix");

      //if (fx_param_exists("do_pca")) {}

      mlpack::IO::StartTimer("kdtree/make_tree");
      *tree_pp = MakeKdTreeMidpoint<TKdTree>(
          matrix, leaflen, old_from_new);
      mlpack::IO::StopTimer("kdtree/make_tree");
    }
    mlpack::IO::StopTimer("kdtree/load");

    return success;
  }


/** Basic KD tree structure. @experimental */
typedef BinarySpaceTree<DHrectBound<2>, Matrix> BasicKdTree;

}; // namespace tree
}; // namespace mlpack

#endif
