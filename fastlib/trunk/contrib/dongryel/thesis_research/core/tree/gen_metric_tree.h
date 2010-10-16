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
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"

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
 *
 * @param metric_in the metric to be used.
 * @param matrix data where each column is a point, WHICH WILL BE RE-ORDERED
 * @param leaf_size the maximum points in a leaf
 * @param max_num_leaf_nodes the number of maximum leaf nodes this tree should
 *        have.
 * @param old_from_new pointer to an unitialized vector; it will map
 *        new indices to original
 * @param new_from_old pointer to an unitialized vector; it will map
 *        original indexes to new indices
 * @param num_nodes the number of nodes constructed in total.
 */
template<typename TMetricTree>
TMetricTree *MakeGenMetricTree(
  const core::metric_kernels::AbstractMetric &metric_in,
  core::table::DenseMatrix& matrix, int leaf_size,
  int max_num_leaf_nodes = std::numeric_limits<int>::max(),
  std::vector<int> *old_from_new = NULL,
  std::vector<int> *new_from_old = NULL,
  int *num_nodes = NULL,
  core::table::MemoryMappedFile *m_file_in = NULL) {

  TMetricTree *node = (m_file_in) ?
                      (TMetricTree *) m_file_in->Allocate(sizeof(TMetricTree)) :
                      new TMetricTree();
  std::vector<int> *old_from_new_ptr;

  if(old_from_new) {
    old_from_new->resize(matrix.n_cols);

    for(unsigned int i = 0; i < matrix.n_cols; i++) {
      (*old_from_new)[i] = i;
    }
    old_from_new_ptr = old_from_new;
  }
  else {
    old_from_new_ptr = NULL;
  }

  int num_nodes_in = 1;
  node->Init(0, matrix.n_cols);
  node->bound().center().Init(matrix.n_rows);
  int current_num_leaf_nodes = 1;
  core::tree_private::SplitGenMetricTree<TMetricTree>(
    metric_in, matrix, node, leaf_size, max_num_leaf_nodes,
    &current_num_leaf_nodes, old_from_new_ptr, &num_nodes_in, m_file_in);

  if(num_nodes) {
    *num_nodes = num_nodes_in;
  }
  if(new_from_old) {
    new_from_old->resize(matrix.n_cols);
    for(unsigned int i = 0; i < matrix.n_cols; i++) {
      (*new_from_old)[(*old_from_new)[i]] = i;
    }
  }
  return node;
}
};
};

#endif
