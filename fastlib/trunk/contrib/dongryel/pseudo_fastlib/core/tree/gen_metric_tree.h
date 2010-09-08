/** @file gen_metric.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_METRIC_TREE_H
#define CORE_TREE_GEN_METRIC_TREE_H

#include <vector>
#include "bounds.h"
#include "general_spacetree.h"
#include "gen_metric_tree_impl.h"

/**
 * Regular pointer-style trees.
 */
namespace core {
namespace tree {

/**
 * Creates a ball tree from data.
 *
 * This requires you to pass in two unitialized ArrayLists which will contain
 * index mappings so you can account for the re-ordering of the matrix.
 * (By unitialized I mean don't call Init on it)
 *
 * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
 * @param leaf_size the maximum points in a leaf
 * @param old_from_new pointer to an unitialized arraylist; it will map
 *        new indices to original
 * @param new_from_old pointer to an unitialized arraylist; it will map
 *        original indexes to new indices
 */
template<typename TMetricTree>
TMetricTree *MakeGenMetricTree(
  arma::mat& matrix, int leaf_size,
  std::vector<int> *old_from_new = NULL,
  std::vector<int> *new_from_old = NULL) {

  TMetricTree *node = new TMetricTree();
  std::vector<int> *old_from_new_ptr;

  if (old_from_new) {
    old_from_new->resize(matrix.n_cols);

    for (int i = 0; i < matrix.n_cols; i++) {
      (*old_from_new)[i] = i;
    }
    old_from_new_ptr = old_from_new;
  }
  else {
    old_from_new_ptr = NULL;
  }

  node->Init(0, matrix.n_cols);
  node->bound().center().set_size(matrix.n_rows);
  core::tree_private::SplitGenMetricTree<TMetricTree>(
    matrix, node, leaf_size, old_from_new_ptr);

  if (new_from_old) {
    new_from_old->resize(matrix.n_cols);
    for (int i = 0; i < matrix.n_cols; i++) {
      (*new_from_old)[(*old_from_new)[i]] = i;
    }
  }

  return node;
}
};
};

#endif
