/** @file matrix_factorized_fmm.h
 *
 *  This file implements a prototype algorithm for computing the
 *  pairwise summation using a matrix-factorized formulation of fast
 *  multipole methods.
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug In progress
 */

#ifndef MATRIX_FACTORIZED_FMM_H
#define MATRIX_FACTORIZED_FMM_H

#include "mlpack/series_expansion/matrix_factorized_farfield_expansion.h"
#include "mlpack/series_expansion/matrix_factorized_local_expansion.h"
#include "fastlib/fastlib.h"

#define INSIDE_MATRIX_FACTORIZED_FMM_IMPL_H



template<typename TKernelAux>
class MatrixFactorizedFMM {
  
 private:
 
#include "matrix_factorized_fmm_stat.h"

  ////////// Private Member Variables //////////

  /** @brief The module holding the parameters.
   */
  struct datanode *module_;

  /** @brief Series expansion auxilary object.
   */
  TKernelAux ka_;
  
  /** @brief The reference dataset.
   */
  Matrix reference_set_;

  /** @brief The reference weights.
   */
  Vector reference_weights_;

  /** @brief The root of the reference tree.
   */
  ReferenceTree *reference_tree_root_;
  
  /** @brief The list of leaf nodes in the reference tree.
   */
  ArrayList<ReferenceTree *> reference_leaf_nodes_;

  /** @brief The permutation mapping indices of reference_set_ to its
   *         original order.
   */
  ArrayList<index_t> old_from_new_references_;

  ////////// Private Member Functions //////////
  
  /** @brief The exhaustive base case for evaluating the reference
   *         contributions to the given set of query points.
   */
  void BaseCase_(const Matrix &query_set, 
		 const ArrayList<index_t> &query_index_permutation,
		 const QueryTree *query_node, 
		 const ReferenceTree *reference_node, 
		 Vector &query_kernel_sums) const;
  
  /** @brief The canonical case for evaluating the reference
   *         contributions to the given set of query points using the
   *         dual-tree algorithm.
   */
  void CanonicalCase_(const Matrix &query_set,
		      const ArrayList<index_t> &query_index_permutation,
		      const QueryTree *query_node,
		      const ReferenceTree *reference_node,
		      Vector &query_kernel_sums) const;

  /** @brief Traverse the FASTLib tree to get the list of leaf nodes.
   */
  template<typename Tree>
  void GetLeafNodes_(Tree *node, ArrayList<Tree *> &leaf_nodes);

  /** @brief The method for preprocessing the query tree.
   */
  void PreProcessQueryTree_
  (const Matrix &query_set, QueryTree *query_node, 
   const Matrix &reference_set, 
   const ArrayList<ReferenceTree *> &reference_leaf_nodes);

  /** @brief The method for preprocessing the reference tree.
   */
  void PreProcessReferenceTree_
  (ReferenceTree *reference_node, const Matrix &query_set, 
   const ArrayList<QueryTree *> &query_leaf_nodes);

 public:
  
  /** @brief Initializes the fast multipole method with the given
   *         reference set.
   *
   *  @param references The reference set.
   *  @param module_in The module holding the parameters.
   */
  void Init(const Matrix &references, struct datanode *module_in);

  /** @brief Compute the weighted kernel sums at each point in the
   *         given query set.
   */
  void Compute(const Matrix &queries, Vector *query_kernel_sums);

};

#include "matrix_factorized_fmm_impl.h"
#undef INSIDE_MATRIX_FACTORIZED_FMM_IMPL_H

#endif
