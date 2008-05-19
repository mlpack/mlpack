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

 public:
  
  /** @brief Initializes the fast multipole method with the given
   *         reference set.
   *
   *  @param references The reference set.
   *  @param module_in The module holding the parameters.
   */
  void Init(const Matrix &references, struct datanode *module_in);

};

#include "matrix_factorized_fmm_impl.h"
#undef INSIDE_MATRIX_FACTORIZED_FMM_IMPL_H

#endif
