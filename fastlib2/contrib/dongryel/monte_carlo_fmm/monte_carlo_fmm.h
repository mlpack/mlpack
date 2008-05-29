/** @file monte_carlo_fmm.h
 *
 *  This file implements a prototype algorithm for computing the
 *  pairwise summation using Monte Carlo sampling.
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug In progress
 */

#ifndef MONTE_CARLO_FMM_H
#define MONTE_CARLO_FMM_H

#define INSIDE_MONTE_CARLO_FMM_H

#include "contrib/dongryel/proximity_project/general_spacetree.h"
#include "contrib/dongryel/proximity_project/gen_metric_tree.h"
#include "mlpack/series_expansion/bounds_aux.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "fastlib/fastlib.h"
#include "monte_carlo_fmm_stat.h"
#include "inverse_normal_cdf.h"

template<typename TKernelAux>
class MonteCarloFMM {
  
 private:

  ////////// Private Constants //////////
  static const int num_initial_samples_per_query_ = 25;

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

  /** @brief The relative error accuracy requirement.
   */
  double relative_error_;

  /** @brief The number of prunes.
   */
  int num_prunes_;

  ////////// Private Member Functions //////////
  
  /** @brief Monte Carlo sampling for determining the prunability of
   *         the given query and the reference node.
   */
  bool MonteCarloPrunable_
  (const Matrix &query_set, const ArrayList<index_t> &query_index_permutation,
   QueryTree *query_node, const ReferenceTree *reference_node,
   Vector &query_kernel_sums, Vector &query_kernel_sums_scratch_space,
   Vector &query_squared_kernel_sums_scratch_space,
   double one_sided_probability);

  /** @brief The exhaustive base case for evaluating the reference
   *         contributions to the given set of query points.
   */
  void BaseCase_(const Matrix &query_set, 
		 const ArrayList<index_t> &query_index_permutation,
		 QueryTree *query_node, const ReferenceTree *reference_node, 
		 Vector &query_kernel_sums) const;
  
  /** @brief The canonical case for evaluating the reference
   *         contributions to the given set of query points using the
   *         dual-tree algorithm.
   */
  void CanonicalCase_(const Matrix &query_set,
		      const ArrayList<index_t> &query_index_permutation,
		      QueryTree *query_node, ReferenceTree *reference_node,
		      Vector &query_kernel_sums, 
		      Vector &query_kernel_sums_scratch_space,
		      Vector &query_squared_kernel_sums_scratch_space,
		      double one_sided_probability);

  /** @brief The method for postprocessing the query tree such that
   *         unclaimed kernel sums are incorporated.
   */
  void PostProcessQueryTree_(const Matrix &query_set, 
			     const ArrayList<index_t> &query_index_permutation,
			     QueryTree *query_node,
			     Vector &query_kernel_sums) const;

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

#include "monte_carlo_fmm_impl.h"
#undef INSIDE_MONTE_CARLO_FMM_H

#endif
