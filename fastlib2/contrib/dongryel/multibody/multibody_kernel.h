#ifndef MULTIBODY_KERNEL_H
#define MULTIBODY_KERNEL_H

#include "fastlib/fastlib.h"
#include "mlpack/series_expansion/kernel_aux.h"

template<typename TTree, typename TBound>
class AxilrodTellerForceKernel {
  
 public:

  ////////// Private Member Constants //////////

  /** @brief The "nu" constant in front of the potential.
   */
  static const double AXILROD_TELLER_COEFF = -91;

 private:

  ////////// Private Member Variables //////////

  /** @brief The temporary matrix to store pairwise distances.
   */
  Matrix distmat_;
  
  /** @brief The temporary ArrayList to store the mapped indices for
   *         computing the gradient.
   */
  ArrayList<index_t> index_orders_;
  
  ////////// Private Member Functions //////////

  double BinomialCoefficient_(double n, double k) {

    double n_k = n - k;
    double nchsk = 1;
    double i;

    if(k > n || k < 0) {
      return 0;
    }
    
    if(k < n_k) {
      k = n_k;
      n_k = n - k;
    }
    
    for(i = 1; i <= n_k; i += 1.0) {
      k += 1.0;
      nchsk *= k;
      nchsk /= i;
    }

    return nchsk;
  }

  double BinomialCoefficientHelper_(double n3, double k3,
				    double n1, double k1, 
				    double n2, double k2) {

    double n_k3 = n3 - k3;
    double n_k1 = n1 - k1;
    double n_k2 = n2 - k2;
    double nchsk = 1;
    double i;

    if(k3 > n3 || k3 < 0 || k1 > n1 || k1 < 0 || k2 > n2 || k2 < 0) {
      return 0;
    }
    
    if(k3 < n_k3) {
      k3 = n_k3;
      n_k3 = n3 - k3;
    }
    if(k1 < n_k1) {
      k1 = n_k1;
      n_k1 = n1 - k1;
    }
    if(k2 < n_k2) {
      k2 = n_k2;
      n_k2 = n2 - k2;
    }

    double min_index = std::min(n_k1, n_k2);
    double max_index = std::max(n_k1, n_k2);
    for(i = 1; i <= min_index; i += 1.0) {
      k1 += 1.0;
      k2 += 1.0;
      nchsk *= k1;
      nchsk /= k2;
    }
    for(i = min_index + 1; i <= max_index; i += 1.0) {
      if(n_k1 < n_k2) {
	k2 += 1.0;
	nchsk *= i;
	nchsk /= k2;
      }
      else {
	k1 += 1.0;
	nchsk *= k1;
	nchsk /= i;
      }
    }
    for(i = 1; i <= n_k3; i += 1.0) {
      k3 += 1.0;
      nchsk *= k3;
      nchsk /= i;
    }

    return nchsk;
  }

  /** @brief Computes the L1 norm of a vector. This probably needs to
   *         be part of the LaPack library.
   */
  double L1Norm_(const Vector &v) {
    double length = 0;

    for(index_t i = 0; i < v.length(); i++) {
      length += fabs(v[i]);
    }
    return length;
  }

  void ComputeAdditionalBoundChanges_
  (ArrayList<TTree *> &nodes, double max_negative_gradient1, 
   double min_positive_gradient1, double max_negative_gradient2, 
   double min_positive_gradient2, double max_negative_gradient3, 
   double min_positive_gradient3, 
   double leave_one_out_node_j_count_for_node_i,
   double leave_one_out_node_k_count_for_node_i,
   double leave_one_out_node_i_count_for_node_j,
   double leave_one_out_node_k_count_for_node_j,
   double leave_one_out_node_i_count_for_node_k,
   double leave_one_out_node_j_count_for_node_k,
   double num_jk_pairs, double num_ik_pairs, 
   double num_ij_pairs, double &node_i_additional_negative_gradient1_u,
   double &node_i_additional_positive_gradient1_l,
   double &node_i_additional_l1_norm_negative_gradient2_u,
   double &node_i_additional_l1_norm_positive_gradient2_l,
   double &node_j_additional_negative_gradient1_u,
   double &node_j_additional_positive_gradient1_l,
   double &node_j_additional_l1_norm_negative_gradient2_u,
   double &node_j_additional_l1_norm_positive_gradient2_l,
   double &node_k_additional_negative_gradient1_u,
   double &node_k_additional_positive_gradient1_l,
   double &node_k_additional_l1_norm_negative_gradient2_u,
   double &node_k_additional_l1_norm_positive_gradient2_l) {

    // i-th node bound changes...
    node_i_additional_negative_gradient1_u = 
      num_jk_pairs * (max_negative_gradient1 + max_negative_gradient2);
    node_i_additional_positive_gradient1_l =
      num_jk_pairs * (min_positive_gradient1 + min_positive_gradient2);
    node_i_additional_l1_norm_negative_gradient2_u =
      fabs(leave_one_out_node_k_count_for_node_i * 
	   (nodes[1]->stat().l1_norm_coordinate_sum_l_) * 
	   max_negative_gradient1 +
	   leave_one_out_node_j_count_for_node_i * 
	   (nodes[2]->stat().l1_norm_coordinate_sum_l_) * 
	   max_negative_gradient2);
    node_i_additional_l1_norm_positive_gradient2_l = 
      leave_one_out_node_k_count_for_node_i * 
      (nodes[1]->stat().l1_norm_coordinate_sum_l_) * min_positive_gradient1 +
      leave_one_out_node_j_count_for_node_i * 
      (nodes[2]->stat().l1_norm_coordinate_sum_l_) * min_positive_gradient2;

    // j-th node bound changes: comptute if any only if it is not the
    // same as the i-th node.
    if(nodes[1] != nodes[0]) {
      node_j_additional_negative_gradient1_u = 
	num_ik_pairs * (max_negative_gradient1 + max_negative_gradient3);
      node_j_additional_positive_gradient1_l =
	num_ik_pairs * (min_positive_gradient1 + min_positive_gradient3);
      node_j_additional_l1_norm_negative_gradient2_u =
	fabs(leave_one_out_node_k_count_for_node_j * 
	     (nodes[0]->stat().l1_norm_coordinate_sum_l_) * 
	     max_negative_gradient1 +
	     leave_one_out_node_i_count_for_node_j * 
	     (nodes[2]->stat().l1_norm_coordinate_sum_l_) * 
	     max_negative_gradient3);
      node_j_additional_l1_norm_positive_gradient2_l = 
	leave_one_out_node_k_count_for_node_j * 
	(nodes[0]->stat().l1_norm_coordinate_sum_l_) * min_positive_gradient1 +
	leave_one_out_node_i_count_for_node_j * 
	(nodes[2]->stat().l1_norm_coordinate_sum_l_) * min_positive_gradient3;
    }
    else {
      node_j_additional_negative_gradient1_u = 
	node_i_additional_negative_gradient1_u;
      node_j_additional_positive_gradient1_l =
	node_i_additional_positive_gradient1_l;
      node_j_additional_l1_norm_negative_gradient2_u =
	node_i_additional_l1_norm_negative_gradient2_u;
      node_j_additional_l1_norm_positive_gradient2_l = 
	node_i_additional_l1_norm_positive_gradient2_l;
    }

    // k-th node bound changes: compute if any only if it is not the
    // same as the j-th node.
    if(nodes[2] != nodes[1]) {
      node_k_additional_negative_gradient1_u = 
	num_ij_pairs * (max_negative_gradient2 + max_negative_gradient3);
      node_k_additional_positive_gradient1_l =
	num_ij_pairs * (min_positive_gradient2 + min_positive_gradient3);
      node_k_additional_l1_norm_negative_gradient2_u =
	fabs(leave_one_out_node_j_count_for_node_k * 
	     (nodes[0]->stat().l1_norm_coordinate_sum_l_) * 
	     max_negative_gradient2 +
	     leave_one_out_node_i_count_for_node_k * 
	     (nodes[1]->stat().l1_norm_coordinate_sum_l_) * 
	     max_negative_gradient3);
      node_k_additional_l1_norm_positive_gradient2_l = 
	leave_one_out_node_j_count_for_node_k * 
	(nodes[0]->stat().l1_norm_coordinate_sum_l_) * min_positive_gradient2 +
	leave_one_out_node_i_count_for_node_k * 
	(nodes[1]->stat().l1_norm_coordinate_sum_l_) * min_positive_gradient3;
    }
    else {
      node_k_additional_negative_gradient1_u = 
	node_j_additional_negative_gradient1_u;
      node_k_additional_positive_gradient1_l =
	node_j_additional_positive_gradient1_l;
      node_k_additional_l1_norm_negative_gradient2_u =
	node_j_additional_l1_norm_negative_gradient2_u;
      node_k_additional_l1_norm_positive_gradient2_l = 
	node_j_additional_l1_norm_positive_gradient2_l;
    }
  }

  void ComputeCurrentAverages_
  (double negative_gradient1_sum, double positive_gradient1_sum,
   double negative_gradient2_sum, double positive_gradient2_sum,
   double negative_gradient3_sum, double positive_gradient3_sum,
   double current_num_samples, double &negative_gradient1_avg,
   double &positive_gradient1_avg, double &negative_gradient2_avg,
   double &positive_gradient2_avg, double &negative_gradient3_avg,
   double &positive_gradient3_avg) {
    
    negative_gradient1_avg = negative_gradient1_sum / 
      ((double) current_num_samples);
    positive_gradient1_avg = positive_gradient1_sum /
      ((double) current_num_samples);
    negative_gradient2_avg = negative_gradient2_sum / 
      ((double) current_num_samples);
    positive_gradient2_avg = positive_gradient2_sum /
      ((double) current_num_samples);
    negative_gradient3_avg = negative_gradient3_sum /
      ((double) current_num_samples);
    positive_gradient3_avg = positive_gradient3_sum /
      ((double) current_num_samples);    
  }

  /** @brief Prune the 3 node tuple based on the lower and the upper
   *         bound approximation. For the Monte Carlo approximation,
   *         lower and upper bound is set to the same sampled mean
   *         quantity.
   */
  void Prune_(ArrayList<TTree *> &tree_nodes, 
	      double negative_gradient1_error, double positive_gradient1_error,
	      double negative_gradient2_error, double positive_gradient2_error,
	      double negative_gradient3_error, double positive_gradient3_error,
	      double min_negative_gradient1,
	      double max_negative_gradient1, double min_positive_gradient1,
	      double max_positive_gradient1, double min_negative_gradient2,
	      double max_negative_gradient2, double min_positive_gradient2,
	      double max_positive_gradient2, double min_negative_gradient3,
	      double max_negative_gradient3, double min_positive_gradient3,
	      double max_positive_gradient3, double num_jk_pairs,
	      double num_ik_pairs, double num_ij_pairs) {

    // First the i-th node.
    tree_nodes[0]->stat().postponed_negative_gradient1_e += 
      num_jk_pairs * 0.5 * (min_negative_gradient1 + max_negative_gradient1 +
			    min_negative_gradient2 + max_negative_gradient2);
    tree_nodes[0]->stat().postponed_negative_gradient1_u += 
      num_jk_pairs * (max_negative_gradient1 + max_negative_gradient2);
    tree_nodes[0]->stat().postponed_positive_gradient1_l += 
      num_jk_pairs * (min_positive_gradient1 + min_positive_gradient2);
    tree_nodes[0]->stat().postponed_positive_gradient1_e += 
      num_jk_pairs * 0.5 * (min_positive_gradient1 + max_positive_gradient1 +
			    min_positive_gradient2 + max_positive_gradient2);
    la::AddExpert
      (tree_nodes[2]->count() *
       0.5 * (min_negative_gradient1 + max_negative_gradient1),
       tree_nodes[1]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_negative_gradient2_e));
    la::AddExpert
      (tree_nodes[1]->count() *
       0.5 * (min_negative_gradient2 + max_negative_gradient2),
       tree_nodes[2]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_negative_gradient2_e));
    la::AddExpert
      (tree_nodes[2]->count() *
       max_negative_gradient1, tree_nodes[1]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_negative_gradient2_u));
    la::AddExpert
      (tree_nodes[1]->count() *
       max_negative_gradient2, tree_nodes[2]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_negative_gradient2_u));
    la::AddExpert
      (tree_nodes[2]->count() *
       min_positive_gradient1, tree_nodes[1]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_positive_gradient2_l));
    la::AddExpert
      (tree_nodes[1]->count() *
       min_positive_gradient2, tree_nodes[2]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_positive_gradient2_l));
    la::AddExpert
      (tree_nodes[2]->count() *
       0.5 * (min_positive_gradient1 + max_positive_gradient1),
       tree_nodes[1]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_positive_gradient2_e));
    la::AddExpert
      (tree_nodes[1]->count() *
       0.5 * (min_positive_gradient2 + max_positive_gradient2),
       tree_nodes[2]->stat().coordinate_sum_,
       &(tree_nodes[0]->stat().postponed_positive_gradient2_e));

    tree_nodes[0]->stat().postponed_negative_gradient1_used_error += 
      num_jk_pairs * (negative_gradient1_error + negative_gradient2_error);
    tree_nodes[0]->stat().postponed_positive_gradient1_used_error += 
      num_jk_pairs * (positive_gradient1_error + positive_gradient2_error);
    tree_nodes[0]->stat().postponed_negative_gradient2_used_error += 
      tree_nodes[2]->count() * tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
      negative_gradient1_error +
      tree_nodes[1]->count() * tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
      negative_gradient2_error;
    tree_nodes[0]->stat().postponed_positive_gradient2_used_error += 
      tree_nodes[2]->count() * tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
      positive_gradient1_error +
      tree_nodes[1]->count() * tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
      positive_gradient2_error;

    // Increment the number of pruned (n - 1) tuples.
    tree_nodes[0]->stat().postponed_n_pruned_ += num_jk_pairs;
   
    // Then the j-th node, if it is not the same as the i-th node...
    if(tree_nodes[1] != tree_nodes[0]) {
      tree_nodes[1]->stat().postponed_negative_gradient1_e += 
	num_ik_pairs * 0.5 * (min_negative_gradient1 + max_negative_gradient1 +
			      min_negative_gradient3 + max_negative_gradient3);
      tree_nodes[1]->stat().postponed_negative_gradient1_u += 
	num_ik_pairs * (max_negative_gradient1 + max_negative_gradient3);
      tree_nodes[1]->stat().postponed_positive_gradient1_l += 
	num_ik_pairs * (min_positive_gradient1 + min_positive_gradient3);
      tree_nodes[1]->stat().postponed_positive_gradient1_e += 
	num_ik_pairs * 0.5 * (min_positive_gradient1 + max_positive_gradient1 +
			      min_positive_gradient3 + max_positive_gradient3);
      la::AddExpert
	(tree_nodes[2]->count() *
	 0.5 * (min_negative_gradient1 + max_negative_gradient1),
	 tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_negative_gradient2_e));
      la::AddExpert
	(tree_nodes[0]->count() *
	 0.5 * (min_negative_gradient3 + max_negative_gradient3),
	 tree_nodes[2]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_negative_gradient2_e));
      la::AddExpert
	(tree_nodes[2]->count() *
	 max_negative_gradient1, tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_negative_gradient2_u));
      la::AddExpert
	(tree_nodes[0]->count() *
	 max_negative_gradient3, tree_nodes[2]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_negative_gradient2_u));
      la::AddExpert
	(tree_nodes[2]->count() *
	 min_positive_gradient1, tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_positive_gradient2_l));
      la::AddExpert
	(tree_nodes[0]->count() *
	 min_positive_gradient3, tree_nodes[2]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_positive_gradient2_l));
      la::AddExpert
	(tree_nodes[2]->count() *
	 0.5 * (min_positive_gradient1 + max_positive_gradient1),
	 tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_positive_gradient2_e));
      la::AddExpert
	(tree_nodes[0]->count() *
	 0.5 * (min_positive_gradient3 + max_positive_gradient3),
	 tree_nodes[2]->stat().coordinate_sum_,
	 &(tree_nodes[1]->stat().postponed_positive_gradient2_e));

      tree_nodes[1]->stat().postponed_negative_gradient1_used_error += 
	num_ik_pairs * (negative_gradient1_error + negative_gradient3_error);
      tree_nodes[1]->stat().postponed_positive_gradient1_used_error += 
	num_ik_pairs * (negative_gradient1_error + negative_gradient3_error);
      tree_nodes[1]->stat().postponed_negative_gradient2_used_error += 
	tree_nodes[2]->count() * 
	tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
	negative_gradient1_error + tree_nodes[0]->count() * 
	tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
	negative_gradient3_error;
      tree_nodes[1]->stat().postponed_positive_gradient2_used_error +=
	tree_nodes[2]->count() * 
	tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
	positive_gradient1_error + tree_nodes[0]->count() * 
	tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
	positive_gradient3_error;
      tree_nodes[1]->stat().postponed_n_pruned_ += num_ik_pairs;
    }

    // Then the k-th node.
    if(tree_nodes[2] != tree_nodes[1]) {
      tree_nodes[2]->stat().postponed_negative_gradient1_e += 
	num_ij_pairs * 0.5 * (min_negative_gradient2 + max_negative_gradient2 +
			      min_negative_gradient3 + max_negative_gradient3);
      tree_nodes[2]->stat().postponed_negative_gradient1_u += 
	num_ij_pairs * (max_negative_gradient2 + max_negative_gradient3);
      tree_nodes[2]->stat().postponed_positive_gradient1_l += 
	num_ij_pairs * (min_positive_gradient2 + min_positive_gradient3);
      tree_nodes[2]->stat().postponed_positive_gradient1_e += 
	num_ij_pairs * 0.5 * (min_positive_gradient2 + max_positive_gradient2 +
			      min_positive_gradient3 + max_positive_gradient3);
      la::AddExpert
	(tree_nodes[1]->count() *
	 0.5 * (min_negative_gradient2 + max_negative_gradient2),
	 tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_negative_gradient2_e));
      la::AddExpert
	(tree_nodes[0]->count() *
	 0.5 * (min_negative_gradient3 + max_negative_gradient3),
	 tree_nodes[1]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_negative_gradient2_e));
      la::AddExpert
	(tree_nodes[1]->count() *
	 max_negative_gradient2, tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_negative_gradient2_u));
      la::AddExpert
	(tree_nodes[0]->count() *
	 max_negative_gradient3, tree_nodes[1]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_negative_gradient2_u));
      la::AddExpert
	(tree_nodes[1]->count() *
	 min_positive_gradient2, tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_positive_gradient2_l));
      la::AddExpert
	(tree_nodes[0]->count() *
	 min_positive_gradient3, tree_nodes[1]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_positive_gradient2_l));
      la::AddExpert
	(tree_nodes[1]->count() *
	 0.5 * (min_positive_gradient2 + max_positive_gradient2),
	 tree_nodes[0]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_positive_gradient2_e));
      la::AddExpert
	(tree_nodes[0]->count() *
	 0.5 * (min_positive_gradient3 + max_positive_gradient3),
	 tree_nodes[1]->stat().coordinate_sum_,
	 &(tree_nodes[2]->stat().postponed_positive_gradient2_e));

      tree_nodes[2]->stat().postponed_negative_gradient1_used_error +=
	num_ij_pairs * (negative_gradient2_error + negative_gradient3_error);
      tree_nodes[2]->stat().postponed_positive_gradient1_used_error += 
	num_ij_pairs * (positive_gradient2_error + positive_gradient3_error);
      tree_nodes[2]->stat().postponed_negative_gradient2_used_error += 
	tree_nodes[1]->count() * 
	tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
	negative_gradient2_error + tree_nodes[0]->count() * 
	tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
	negative_gradient3_error;
      tree_nodes[2]->stat().postponed_positive_gradient2_used_error += 
	tree_nodes[1]->count() * 
	tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
	positive_gradient2_error + tree_nodes[0]->count() * 
	tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
	positive_gradient3_error;

      tree_nodes[2]->stat().postponed_n_pruned_ += num_ij_pairs;
    }
  }

  /** @brief Determine whether the 3-node tuple can be pruned based on
   *         bounds.
   */
  bool Prunable_(ArrayList<TTree *> &tree_nodes,
		 double negative_gradient1_error, 
		 double positive_gradient1_error,
		 double negative_gradient2_error,
		 double positive_gradient2_error,
		 double negative_gradient3_error,
		 double positive_gradient3_error,
		 double node_i_additional_negative_force1_u,
		 double node_i_additional_positive_force1_l,
		 double node_i_additional_l1_norm_negative_force2_u,
		 double node_i_additional_l1_norm_positive_force2_l,
		 double node_j_additional_negative_force1_u,
		 double node_j_additional_positive_force1_l,
		 double node_j_additional_l1_norm_negative_force2_u,
		 double node_j_additional_l1_norm_positive_force2_l,
		 double node_k_additional_negative_force1_u,
		 double node_k_additional_positive_force1_l,
		 double node_k_additional_l1_norm_negative_force2_u,
		 double node_k_additional_l1_norm_positive_force2_l,
		 double num_jk_pairs, double num_ik_pairs, 
		 double num_ij_pairs, double relative_error,
		 double threshold,
		 double total_n_minus_one_num_tuples) {

    bool first_node_prunable =
      (num_jk_pairs *
       (negative_gradient1_error + negative_gradient2_error) <=
       std::max
       ((relative_error *
	 fabs(tree_nodes[0]->stat().negative_gradient1_u +
	      tree_nodes[0]->stat().postponed_negative_gradient1_u +
	      node_i_additional_negative_force1_u) -
	 tree_nodes[0]->stat().negative_gradient1_used_error) *
	(num_jk_pairs / 
	 (total_n_minus_one_num_tuples - tree_nodes[0]->stat().n_pruned_)),
	threshold * (num_jk_pairs / total_n_minus_one_num_tuples)))
      &&
      (num_jk_pairs *
       (positive_gradient1_error + positive_gradient2_error) <=
       std::max
       ((relative_error *
	(tree_nodes[0]->stat().positive_gradient1_l +
	 tree_nodes[0]->stat().postponed_positive_gradient1_l +
	 node_i_additional_positive_force1_l) -
	 tree_nodes[0]->stat().positive_gradient1_used_error) *
	(num_jk_pairs /
	 (total_n_minus_one_num_tuples - tree_nodes[0]->stat().n_pruned_)),
	threshold / (num_jk_pairs / total_n_minus_one_num_tuples)))
      &&
      (tree_nodes[2]->count() * tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
       negative_gradient1_error +
       tree_nodes[1]->count() * tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
       negative_gradient2_error <=
       std::max
       ((relative_error *
	 (tree_nodes[0]->stat().l1_norm_negative_gradient2_u +
	  L1Norm_(tree_nodes[0]->stat().postponed_negative_gradient2_u) +
	  node_i_additional_l1_norm_negative_force2_u) -
	 tree_nodes[0]->stat().negative_gradient2_used_error) *
	(num_jk_pairs / 
	 (total_n_minus_one_num_tuples - tree_nodes[0]->stat().n_pruned_)),
	threshold * (num_jk_pairs / total_n_minus_one_num_tuples)))
      &&
      (tree_nodes[2]->count() * tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
       positive_gradient1_error +
       tree_nodes[1]->count() * tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
       positive_gradient2_error <=
       std::max
       ((relative_error * 
	 (tree_nodes[0]->stat().l1_norm_positive_gradient2_l +
	  L1Norm_(tree_nodes[0]->stat().postponed_positive_gradient2_l) +
	  node_i_additional_l1_norm_positive_force2_l) -
	 tree_nodes[0]->stat().positive_gradient2_used_error) *
	(num_jk_pairs / 
	 (total_n_minus_one_num_tuples - tree_nodes[0]->stat().n_pruned_)),
	threshold * (num_jk_pairs / total_n_minus_one_num_tuples)));
    
    // Short circuit prunable decision...
    if(!first_node_prunable) {
      return false;
    }

    bool second_node_prunable =
      (tree_nodes[1] == tree_nodes[0]) ? first_node_prunable:
      ((num_ik_pairs *
	(negative_gradient1_error + negative_gradient3_error) <=
	std::max
	((relative_error *
	  fabs(tree_nodes[1]->stat().negative_gradient1_u +
	       tree_nodes[1]->stat().postponed_negative_gradient1_u +
	       node_j_additional_negative_force1_u) -
	  tree_nodes[1]->stat().negative_gradient1_used_error) *
	 (num_ik_pairs / 
	  (total_n_minus_one_num_tuples - tree_nodes[1]->stat().n_pruned_)),
	 threshold * (num_ik_pairs / total_n_minus_one_num_tuples)))
       &&
       (num_ik_pairs *
	(positive_gradient1_error + positive_gradient3_error) <=
	std::max
	((relative_error *
	  (tree_nodes[1]->stat().positive_gradient1_l +
	   tree_nodes[1]->stat().postponed_positive_gradient1_l +
	   node_j_additional_positive_force1_l) -
	  tree_nodes[1]->stat().positive_gradient1_used_error) *
	 (num_ik_pairs /
	  (total_n_minus_one_num_tuples - tree_nodes[1]->stat().n_pruned_)),
	 threshold * (num_ik_pairs / total_n_minus_one_num_tuples)))
       &&
       (tree_nodes[2]->count() * 
	tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
	negative_gradient1_error +
	tree_nodes[0]->count() * 
	tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
	negative_gradient3_error <=
	std::max
	((relative_error *
	  (tree_nodes[1]->stat().l1_norm_negative_gradient2_u +
	   L1Norm_
	   (tree_nodes[1]->stat().postponed_negative_gradient2_u) +
	   node_j_additional_l1_norm_negative_force2_u) -
	  tree_nodes[1]->stat().negative_gradient2_used_error) *
	 (num_ik_pairs /
	  (total_n_minus_one_num_tuples - tree_nodes[1]->stat().n_pruned_)),
	 threshold * (num_ik_pairs / total_n_minus_one_num_tuples)))
       &&
      (tree_nodes[2]->count() * tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
       positive_gradient1_error +
       tree_nodes[0]->count() * tree_nodes[2]->stat().l1_norm_coordinate_sum_ *
       positive_gradient3_error <=
       std::max
       ((relative_error *
	 (tree_nodes[1]->stat().l1_norm_positive_gradient2_l +
	  L1Norm_
	  (tree_nodes[1]->stat().postponed_positive_gradient2_l) +
	  node_j_additional_l1_norm_positive_force2_l) -
	 tree_nodes[1]->stat().positive_gradient2_used_error) *
	(num_ik_pairs / 
	 (total_n_minus_one_num_tuples - tree_nodes[1]->stat().n_pruned_)),
	threshold * (num_ik_pairs / total_n_minus_one_num_tuples))));

    // Short circuit prunable decision based on the second node result...
    if(!second_node_prunable) {
      return false;
    }

    bool third_node_prunable =
      (tree_nodes[2] == tree_nodes[1]) ? second_node_prunable:
      ((num_ij_pairs *
	(negative_gradient2_error + negative_gradient3_error) <=
	std::max
	((relative_error *
	  fabs(tree_nodes[2]->stat().negative_gradient1_u +
	       tree_nodes[2]->stat().postponed_negative_gradient1_u +
	       node_k_additional_negative_force1_u) -
	  tree_nodes[2]->stat().negative_gradient1_used_error) *
	 (num_ij_pairs / 
	  (total_n_minus_one_num_tuples - tree_nodes[2]->stat().n_pruned_)),
	 threshold * (num_ij_pairs / total_n_minus_one_num_tuples)))
       &&
       (num_ij_pairs *
	(positive_gradient2_error + positive_gradient3_error) <=
	std::max
	((relative_error *
	 (tree_nodes[2]->stat().positive_gradient1_l +
	  tree_nodes[2]->stat().postponed_positive_gradient1_l +
	  node_k_additional_positive_force1_l) -
	  tree_nodes[2]->stat().positive_gradient1_used_error) *
	 (num_ij_pairs /
	  (total_n_minus_one_num_tuples - tree_nodes[2]->stat().n_pruned_)),
	 threshold * (num_ij_pairs / total_n_minus_one_num_tuples)))
       &&
       (tree_nodes[1]->count() * 
	tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
	negative_gradient2_error +
	tree_nodes[0]->count() * 
	tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
	negative_gradient3_error <=
	std::max
	((relative_error *
	  (tree_nodes[2]->stat().l1_norm_negative_gradient2_u +
	   L1Norm_
	   (tree_nodes[2]->stat().postponed_negative_gradient2_u) +
	   node_k_additional_l1_norm_negative_force2_u) -
	  tree_nodes[2]->stat().negative_gradient2_used_error) *
	 (num_ij_pairs /
	  (total_n_minus_one_num_tuples - tree_nodes[2]->stat().n_pruned_)),
	 threshold * (num_ij_pairs / total_n_minus_one_num_tuples)))
       &&
       (tree_nodes[1]->count() * 
	tree_nodes[0]->stat().l1_norm_coordinate_sum_ *
	positive_gradient2_error +
	tree_nodes[0]->count() * 
	tree_nodes[1]->stat().l1_norm_coordinate_sum_ *
	positive_gradient3_error <=
	std::max
	((relative_error *
	  (tree_nodes[2]->stat().l1_norm_positive_gradient2_l +
	   L1Norm_
	   (tree_nodes[2]->stat().postponed_positive_gradient2_l) +
	   node_k_additional_l1_norm_positive_force2_l) -
	  tree_nodes[2]->stat().positive_gradient2_used_error) *
	 (num_ij_pairs / 
	  (total_n_minus_one_num_tuples - tree_nodes[2]->stat().n_pruned_)),
	 threshold * (num_ij_pairs / total_n_minus_one_num_tuples))));
       
    // Prunable if any only if all three nodes are prunable.
    return third_node_prunable;
  }
  

  /** @brief Computes error due to finite difference approximation.
   */
  void ComputeGradientComponentError_
  (double min_negative_gradient1, double max_negative_gradient1,
   double min_positive_gradient1, double max_positive_gradient1,
   double min_negative_gradient2, double max_negative_gradient2,
   double min_positive_gradient2, double max_positive_gradient2,
   double min_negative_gradient3, double max_negative_gradient3,
   double min_positive_gradient3, double max_positive_gradient3,
   double &negative_gradient1_error, double &positive_gradient1_error,
   double &negative_gradient2_error, double &positive_gradient2_error,
   double &negative_gradient3_error, double &positive_gradient3_error) {
    
    negative_gradient1_error = 
      (max_negative_gradient1 - min_negative_gradient1) * 0.5;
    positive_gradient1_error =
      (max_positive_gradient1 - min_positive_gradient1) * 0.5;
    negative_gradient2_error = 
      (max_negative_gradient2 - min_negative_gradient2) * 0.5;
    positive_gradient2_error =
      (max_positive_gradient2 - min_positive_gradient2) * 0.5;
    negative_gradient3_error = 
      (max_negative_gradient3 - min_negative_gradient3) * 0.5;
    positive_gradient3_error =
      (max_positive_gradient3 - min_positive_gradient3) * 0.5;    
  }

  /** @brief Computes error due to Monte Carlo approximation.
   */
  void ComputeMonteCarloGradientComponentError_
  (double negative_gradient1_sum, double negative_gradient1_squared_sum,
   double positive_gradient1_sum, double positive_gradient1_squared_sum,
   double negative_gradient2_sum, double negative_gradient2_squared_sum,
   double positive_gradient2_sum, double positive_gradient2_squared_sum,
   double negative_gradient3_sum, double negative_gradient3_squared_sum,
   double positive_gradient3_sum, double positive_gradient3_squared_sum,
   int num_samples, double z_score,
   double &negative_gradient1_error, double &positive_gradient1_error,
   double &negative_gradient2_error, double &positive_gradient2_error,
   double &negative_gradient3_error, double &positive_gradient3_error) {

    double inverse_factor = 1.0 / ((double) num_samples - 1.0);
    
    // First compute the variance component.
    negative_gradient1_error = 
      inverse_factor * (negative_gradient1_squared_sum - 
			negative_gradient1_sum * negative_gradient1_sum / 
			((double) num_samples));
    positive_gradient1_error =
      inverse_factor * (positive_gradient1_squared_sum -
			positive_gradient1_sum * positive_gradient1_sum /
			((double) num_samples));
    negative_gradient2_error = 
      inverse_factor * (negative_gradient2_squared_sum - 
			negative_gradient2_sum * negative_gradient2_sum / 
			((double) num_samples));
    positive_gradient2_error =
      inverse_factor * (positive_gradient2_squared_sum -
			positive_gradient2_sum * positive_gradient2_sum /
			((double) num_samples));
    negative_gradient3_error = 
      inverse_factor * (negative_gradient3_squared_sum - 
			negative_gradient3_sum * negative_gradient3_sum / 
			((double) num_samples));
    positive_gradient3_error =
      inverse_factor * (positive_gradient3_squared_sum -
			positive_gradient3_sum * positive_gradient3_sum /
			((double) num_samples));

    // Then transform each variance into the actual confidence
    // interval range (for sample variance, not the sample mean
    // variance).
    negative_gradient1_error = z_score * sqrt(negative_gradient1_error);
    positive_gradient1_error = z_score * sqrt(positive_gradient1_error);
    negative_gradient2_error = z_score * sqrt(negative_gradient2_error);
    positive_gradient2_error = z_score * sqrt(positive_gradient2_error);
    negative_gradient3_error = z_score * sqrt(negative_gradient3_error);
    positive_gradient3_error = z_score * sqrt(positive_gradient3_error);    
  }

  void force_(const Matrix &data, const ArrayList<index_t> &indices, 
	      double &negative_gradient1, double &positive_gradient1, 
	      double &negative_gradient2, double &positive_gradient2,
	      double &negative_gradient3, double &positive_gradient3,
	      Vector &negative_force1_e, Vector &negative_force1_u,
	      Vector &positive_force1_l, Vector &positive_force1_e,
	      Matrix &negative_force2_e, Matrix &negative_force2_u,
	      Matrix &positive_force2_l, Matrix &positive_force2_e) {

    // Negative contribution to the first component.
    negative_force1_e[indices[index_orders_[0]]] += 
      negative_gradient1 + negative_gradient2;
    negative_force1_u[indices[index_orders_[0]]] += 
      negative_gradient1 + negative_gradient2;

    // Positive contribution to the first component.
    positive_force1_l[indices[index_orders_[0]]] += 
      positive_gradient1 + positive_gradient2;
    positive_force1_e[indices[index_orders_[0]]] += 
      positive_gradient1 + positive_gradient2;

    // Negative contribution to the second component.
    la::AddExpert(data.n_rows(), negative_gradient1, 
		  data.GetColumnPtr(indices[index_orders_[1]]),
		  negative_force2_e.GetColumnPtr(indices[index_orders_[0]]));
    la::AddExpert(data.n_rows(), negative_gradient2,
		  data.GetColumnPtr(indices[index_orders_[2]]),
		  negative_force2_e.GetColumnPtr(indices[index_orders_[0]]));
    la::AddExpert(data.n_rows(), negative_gradient1, 
		  data.GetColumnPtr(indices[index_orders_[1]]),
		  negative_force2_u.GetColumnPtr(indices[index_orders_[0]]));
    la::AddExpert(data.n_rows(), negative_gradient2,
		  data.GetColumnPtr(indices[index_orders_[2]]),
		  negative_force2_u.GetColumnPtr(indices[index_orders_[0]]));

    // Positive contribution to the second component.
    la::AddExpert(data.n_rows(), positive_gradient1, 
		  data.GetColumnPtr(indices[index_orders_[1]]),
		  positive_force2_e.GetColumnPtr(indices[index_orders_[0]]));
    la::AddExpert(data.n_rows(), positive_gradient2,
		  data.GetColumnPtr(indices[index_orders_[2]]),
		  positive_force2_e.GetColumnPtr(indices[index_orders_[0]]));
    la::AddExpert(data.n_rows(), positive_gradient1, 
		  data.GetColumnPtr(indices[index_orders_[1]]),
		  positive_force2_l.GetColumnPtr(indices[index_orders_[0]]));
    la::AddExpert(data.n_rows(), positive_gradient2,
		  data.GetColumnPtr(indices[index_orders_[2]]),
		  positive_force2_l.GetColumnPtr(indices[index_orders_[0]]));
  }

  void gradient_(const ArrayList<index_t> &index_orders, 
		 double &minimum_negative_gradient,
		 double *maximum_negative_gradient,
		 double &minimum_positive_gradient,
		 double *maximum_positive_gradient) {

    // Between the i-th node and the j-th node.
    int min_index1 = std::min(index_orders[0], index_orders[1]);
    int max_index1 = std::max(index_orders[0], index_orders[1]);
    double min_dsqd1 = distmat_.get(min_index1, max_index1);
    double min_dist1 = sqrt(min_dsqd1);
    double min_dqrt1 = math::Sqr(min_dsqd1);
    double min_dsix1 = min_dsqd1 * min_dqrt1;

    double max_dsqd1 = distmat_.get(max_index1, min_index1);
    double max_dist1 = sqrt(max_dsqd1);
    double max_dqrt1 = math::Sqr(max_dsqd1);
    double max_dsix1 = max_dsqd1 * max_dqrt1;

    // Between the i-th node and the k-th node.
    int min_index2 = std::min(index_orders[0], index_orders[2]);
    int max_index2 = std::max(index_orders[0], index_orders[2]);
    double min_dsqd2 = distmat_.get(min_index2, max_index2);
    double min_dist2 = sqrt(min_dsqd2);
    double min_dcub2 = min_dsqd2 * min_dist2;
    double min_dqui2 = min_dsqd2 * min_dcub2;

    double max_dsqd2 = distmat_.get(max_index2, min_index2);
    double max_dist2 = sqrt(max_dsqd2);
    double max_dcub2 = max_dsqd2 * max_dist2;
    double max_dqui2 = max_dsqd2 * max_dcub2;
    
    // Between the j-th node and the k-th node.
    int min_index3 = std::min(index_orders[1], index_orders[2]);
    int max_index3 = std::max(index_orders[1], index_orders[2]);
    double min_dsqd3 = distmat_.get(min_index3, max_index3);
    double min_dist3 = sqrt(min_dsqd3);
    double min_dcub3 = min_dsqd3 * min_dist3;
    double min_dqui3 = min_dsqd3 * min_dcub3;

    double max_dsqd3 = distmat_.get(max_index3, min_index3);
    double max_dist3 = sqrt(max_dsqd3);
    double max_dcub3 = max_dsqd3 * max_dist3;
    double max_dqui3 = max_dsqd3 * max_dcub3;
    
    // Common factor in front.
    double min_common_factor = 3.0 / (8.0 * max_dist1);
    double max_common_factor = 3.0 / (8.0 * min_dist1);

    minimum_negative_gradient = max_common_factor *
      (-2.0 / (min_dqrt1 * min_dcub2 * min_dcub3)
       - 1.0 / (min_dqui2 * min_dqui3)
       - 1.0 / (min_dsqd1 * min_dcub2 * min_dqui3)
       - 1.0 / (min_dsqd1 * min_dqui2 * min_dcub3)
       - 3.0 / (min_dqrt1 * min_dist2 * min_dqui3)
       - 3.0 / (min_dqrt1 * min_dqui2 * min_dist3)
       - 5.0 / (min_dsix1 * min_dist2 * min_dcub3)
       - 5.0 / (min_dsix1 * min_dcub2 * min_dist3));
    
    if(maximum_negative_gradient) {
      *maximum_negative_gradient = min_common_factor *
	(-2.0 / (max_dqrt1 * max_dcub2 * max_dcub3)
	 - 1.0 / (max_dqui2 * max_dqui3)
	 - 1.0 / (max_dsqd1 * max_dcub2 * max_dqui3)
	 - 1.0 / (max_dsqd1 * max_dqui2 * max_dcub3)
	 - 3.0 / (max_dqrt1 * max_dist2 * max_dqui3)
	 - 3.0 / (max_dqrt1 * max_dqui2 * max_dist3)
	 - 5.0 / (max_dsix1 * max_dist2 * max_dcub3)
	 - 5.0 / (max_dsix1 * max_dcub2 * max_dist3));
    }

    minimum_positive_gradient = min_common_factor *
      (5 * min_dist2 / (max_dsix1 * max_dqui3) +
       5 * min_dist3 / (max_dsix1 * max_dqui2));

    if(maximum_positive_gradient) {
      *maximum_positive_gradient = max_common_factor *
	(5 * max_dist2 / (min_dsix1 * min_dqui3) +
	 5 * max_dist3 / (min_dsix1 * min_dqui2));
    }
  }

 public:
  
  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  AxilrodTellerForceKernel() {
  }

  /** @brief The default destructor.
   */
  ~AxilrodTellerForceKernel() {
  }

  ////////// Getters/Setters //////////
  
  /** @brief Gets the squared distance matrix.
   */
  const Matrix &pairwise_squared_distances() const { return distmat_; }
  
  /** @brief Gets the interaction order of the kernel.
   */
  int order() {
    return 3;
  }

  ////////// User-level Functions //////////

  /** @brief Initializes the kernel.
   */
  void Init(double bandwidth_in) {
    distmat_.Init(3, 3);
    index_orders_.Init(3);
  }

  /** @brief Computes the outer confidence interval for the quantile
   *         intervals.
   */
  double OuterConfidenceInterval
  (double population_size, double sample_size,
   double sample_order_statistics_min_index,
   double population_order_statistics_min_index) {

    double total_probability = 0;
    double lower_percentile = population_order_statistics_min_index /
      population_size;
    
    for(double r_star = sample_order_statistics_min_index;
	r_star <= std::min(population_order_statistics_min_index, sample_size);
	r_star += 1.0) {

      // If any of the arguments to the binomial coefficient is
      // invalid, then the contribution is zero.
      if(r_star > population_order_statistics_min_index ||
	 sample_size - r_star < 0 || 
	 population_size - population_order_statistics_min_index < 0 ||
	 sample_size - r_star >
	 population_size - population_order_statistics_min_index) {
	continue;
      }
      
      /*
      total_probability +=
	BinomialCoefficientHelper_
	(population_order_statistics_min_index, r_star,
	 population_size - population_order_statistics_min_index,
	 sample_size - r_star, population_size, sample_size);
      */
      total_probability +=
	math::BinomialCoefficient((int) sample_size, (int) r_star) *
	pow(lower_percentile, r_star) * 
	pow(1 - lower_percentile, sample_size - r_star);
    }
    return std::max(std::min(total_probability, 1.0), 0.0);
  }

  /** @brief Computes the leave-one-out node count and the 2-tuple
   *         counts.
   *
   *  @param nodes The set of nodes (3 for Axilrod-Teller).
   *  @param leave_one_out_node_j_count_for_node_i
   *  @param leave_one_out_node_k_count_for_node_i
   *  @param leave_one_out_node_i_count_for_node_j
   *  @param leave_one_out_node_k_count_for_node_j
   *  @param leave_one_out_node_i_count_for_node_k
   *  @param leave_one_out_node_j_count_for_node_k
   *  @param num_jk_pairs
   *  @param num_ik_pairs
   *  @param num_ij_pairs
   */
  void ComputeNumTwoTuples_
  (const ArrayList<TTree *> &nodes,
   double &leave_one_out_node_j_count_for_node_i,
   double &leave_one_out_node_k_count_for_node_i,
   double &leave_one_out_node_i_count_for_node_j,
   double &leave_one_out_node_k_count_for_node_j,
   double &leave_one_out_node_i_count_for_node_k,
   double &leave_one_out_node_j_count_for_node_k,
   double &num_jk_pairs, double &num_ik_pairs, double &num_ij_pairs) {
    
    if(nodes[0] == nodes[1]) {

      // All three nodes are equal...
      if(nodes[1] == nodes[2]) {
	leave_one_out_node_j_count_for_node_i = nodes[0]->count() - 1;
	leave_one_out_node_k_count_for_node_i = nodes[0]->count() - 2;
	leave_one_out_node_i_count_for_node_j = nodes[0]->count() - 1;
	leave_one_out_node_k_count_for_node_j = nodes[0]->count() - 2;
	leave_one_out_node_i_count_for_node_k = nodes[0]->count() - 1;
	leave_one_out_node_j_count_for_node_k = nodes[0]->count() - 2;
	num_jk_pairs = math::BinomialCoefficient(nodes[0]->count() - 1, 2);
	num_ik_pairs = num_jk_pairs;
	num_ij_pairs = num_jk_pairs;
      }

      // i-th node equals j-th node, but j-th node does not equal k-th
      // node.
      else {
	leave_one_out_node_j_count_for_node_i = nodes[0]->count() - 1;
	leave_one_out_node_k_count_for_node_i = nodes[2]->count();
	leave_one_out_node_i_count_for_node_j = nodes[0]->count() - 1;
	leave_one_out_node_k_count_for_node_j = nodes[2]->count();
	leave_one_out_node_i_count_for_node_k = nodes[0]->count();
	leave_one_out_node_j_count_for_node_k = nodes[0]->count() - 1;
	num_jk_pairs = (nodes[0]->count() - 1) * (nodes[2]->count());
	num_ik_pairs = num_jk_pairs;
	num_ij_pairs = math::BinomialCoefficient(nodes[0]->count(), 2);	
      }
    }
    else {
      
      // i-th node does not equal j-th node, but j-th node equals k-th
      // node.
      if(nodes[1] == nodes[2]) {
	leave_one_out_node_j_count_for_node_i = nodes[1]->count();
	leave_one_out_node_k_count_for_node_i = nodes[1]->count() - 1;
	leave_one_out_node_i_count_for_node_j = nodes[0]->count();
	leave_one_out_node_k_count_for_node_j = nodes[1]->count() - 1;
	leave_one_out_node_i_count_for_node_k = nodes[0]->count();
	leave_one_out_node_j_count_for_node_k = nodes[1]->count() - 1;
	num_jk_pairs = math::BinomialCoefficient(nodes[1]->count(), 2);
	num_ik_pairs = (nodes[0]->count()) * (nodes[2]->count() - 1);
	num_ij_pairs = (nodes[0]->count()) * (nodes[1]->count() - 1);
      }

      // All three nodes are disjoint in this case...
      else {
	leave_one_out_node_j_count_for_node_i = nodes[1]->count();
	leave_one_out_node_k_count_for_node_i = nodes[2]->count();
	leave_one_out_node_i_count_for_node_j = nodes[0]->count();
	leave_one_out_node_k_count_for_node_j = nodes[2]->count();
	leave_one_out_node_i_count_for_node_k = nodes[0]->count();
	leave_one_out_node_j_count_for_node_k = nodes[1]->count();
	num_jk_pairs = (nodes[1]->count()) * (nodes[2]->count());
	num_ik_pairs = (nodes[0]->count()) * (nodes[2]->count());
	num_ij_pairs = (nodes[0]->count()) * (nodes[1]->count());
      }
    }
  }

  /** @brief Computes the pairwise distance among FastLib tree nodes.
   */
  void EvalMinMaxSquaredDistances(ArrayList<TTree *> &tree_nodes) {

    int num_nodes = tree_nodes.size();

    for(index_t i = 0; i < num_nodes - 1; i++) {
      const TBound &node_i_bound = tree_nodes[i]->bound();

      for(index_t j = i + 1; j < num_nodes; j++) {
	const TBound &node_j_bound = tree_nodes[j]->bound();
        double min_squared_distance = 
	  std::max(node_i_bound.MinDistanceSq(node_j_bound),
		   std::min(tree_nodes[i]->stat().knn_dsqds_lower_bounds_[0],
			    tree_nodes[j]->stat().knn_dsqds_lower_bounds_[0]));
	double max_squared_distance =
	  std::min(node_i_bound.MaxDistanceSq(node_j_bound),
		   std::max(tree_nodes[i]->stat().kfn_dsqds_upper_bounds_[0],
			    tree_nodes[j]->stat().kfn_dsqds_upper_bounds_[0]));

        distmat_.set(i, j, min_squared_distance);
        distmat_.set(j, i, max_squared_distance);
      }
    }
  }

  /** @brief Computes the pairwise distance among three points.
   */
  void EvalMinMaxSquaredDistances(const Matrix &data, 
				  const ArrayList<index_t> &indices) {
    
    int num_order = order();
    
    for(index_t i = 0; i < num_order - 1; i++) {
      
      const double *point_i = data.GetColumnPtr(indices[i]);
      
      for(index_t j = i + 1; j < num_order; j++) {
	const double *point_j = data.GetColumnPtr(indices[j]);
        double squared_distance = la::DistanceSqEuclidean(data.n_rows(), 
							  point_i, point_j);
        distmat_.set(i, j, squared_distance);
        distmat_.set(j, i, squared_distance);
      }
    }
  }

  /** @brief Computes $\frac{\nu}{r_i - r_j} \frac{\partial
   *         u}{\partial (r_i - r_j)}$, $\frac{\nu}{r_i - r_k}
   *         \frac{\partial u}{\partial (r_i - r_k)}$ and
   *         $\frac{\nu}{r_j - r_k} \frac{\partial u}{\partial (r_j -
   *         r_k)}$.
   */
  void EvalGradients(const Matrix &dsqd_matrix,
		     double &min_negative_gradient1,
		     double *max_negative_gradient1,
		     double &min_positive_gradient1, 
		     double *max_positive_gradient1,
		     double &min_negative_gradient2, 
		     double *max_negative_gradient2,
		     double &min_positive_gradient2, 
		     double *max_positive_gradient2,
		     double &min_negative_gradient3, 
		     double *max_negative_gradient3,
		     double &min_positive_gradient3, 
		     double *max_positive_gradient3) {

    index_orders_[0] = 0;
    index_orders_[1] = 1;
    index_orders_[2] = 2;
    gradient_(index_orders_, min_negative_gradient1, max_negative_gradient1,
	      min_positive_gradient1, max_positive_gradient1);

    index_orders_[0] = 0;
    index_orders_[1] = 2;
    index_orders_[2] = 1;
    gradient_(index_orders_, min_negative_gradient2, max_negative_gradient2,
	      min_positive_gradient2, max_positive_gradient2);
    
    index_orders_[0] = 2;
    index_orders_[1] = 1;
    index_orders_[2] = 0;
    gradient_(index_orders_, min_negative_gradient3, max_negative_gradient3,
	      min_positive_gradient3, max_positive_gradient3);
  }

  void EvalContributions
  (const Matrix &data, const ArrayList<index_t> &indices,
   double &negative_gradient1, double &positive_gradient1,
   double &negative_gradient2, double &positive_gradient2,
   double &negative_gradient3, double &positive_gradient3,
   Vector &negative_force1_e, Vector &negative_force1_u,
   Vector &positive_force1_l, Vector &positive_force1_e,
   Matrix &negative_force2_e, Matrix &negative_force2_u,
   Matrix &positive_force2_l, Matrix &positive_force2_e) {
   
    index_orders_[0] = 0;
    index_orders_[1] = 1;
    index_orders_[2] = 2;
    force_(data, indices, negative_gradient1, positive_gradient1, 
	   negative_gradient2, positive_gradient2,
	   negative_gradient3, positive_gradient3,
	   negative_force1_e, negative_force1_u,
	   positive_force1_l, positive_force1_e,
	   negative_force2_e, negative_force2_u,
	   positive_force2_l, positive_force2_e);
    
    index_orders_[0] = 1;
    index_orders_[1] = 0;
    index_orders_[2] = 2;
    force_(data, indices, negative_gradient1, positive_gradient1,
	   negative_gradient3, positive_gradient3,
	   negative_gradient2, positive_gradient2,
	   negative_force1_e, negative_force1_u,
	   positive_force1_l, positive_force1_e,
	   negative_force2_e, negative_force2_u,
	   positive_force2_l, positive_force2_e);
    
    index_orders_[0] = 2;
    index_orders_[1] = 0;
    index_orders_[2] = 1;
    force_(data, indices, negative_gradient2, positive_gradient2, 
	   negative_gradient3, positive_gradient3,
	   negative_gradient1, positive_gradient1,
	   negative_force1_e, negative_force1_u,
	   positive_force1_l, positive_force1_e,
	   negative_force2_e, negative_force2_u,
	   positive_force2_l, positive_force2_e);
  }

  /** @brief Computes the first/second components of the
   *         negative/positive force components.
   */
  void Eval(const Matrix &data, const ArrayList<index_t> &indices,
	    Vector &negative_force1_e, Vector &negative_force1_u,
	    Vector &positive_force1_l, Vector &positive_force1_e,
	    Matrix &negative_force2_e, Matrix &negative_force2_u,
	    Matrix &positive_force2_l, Matrix &positive_force2_e) {

    double negative_gradient1, positive_gradient1;
    double negative_gradient2, positive_gradient2;
    double negative_gradient3, positive_gradient3;
    
    // Evaluate the pairwise distances among all points.
    EvalMinMaxSquaredDistances(data, indices);

    // Evaluate the required components of the force vector.
    EvalGradients(distmat_, negative_gradient1, NULL, positive_gradient1, NULL,
		  negative_gradient2, NULL, positive_gradient2, NULL,
		  negative_gradient3, NULL, positive_gradient3, NULL);

    // Contributions to all three particles in the list.
    EvalContributions(data, indices,
		      negative_gradient1, positive_gradient1,
		      negative_gradient2, positive_gradient2,
		      negative_gradient3, positive_gradient3,
		      negative_force1_e, negative_force1_u,
		      positive_force1_l, positive_force1_e,
		      negative_force2_e, negative_force2_u,
		      positive_force2_l, positive_force2_e);
  }

  void UpdateStatistics_
  (double negative_gradient1, double positive_gradient1,
   double negative_gradient2, double positive_gradient2,
   double negative_gradient3, double positive_gradient3,
   double &min_negative_gradient1, double &min_positive_gradient1,
   double &min_negative_gradient2, double &min_positive_gradient2,
   double &min_negative_gradient3, double &min_positive_gradient3) {

    min_negative_gradient1 = std::min(min_negative_gradient1,
				      negative_gradient1);
    min_positive_gradient1 = std::max(min_positive_gradient1,
				      positive_gradient1);
    
    min_negative_gradient2 = std::min(min_negative_gradient2,
				      negative_gradient2);
    min_positive_gradient2 = std::max(min_positive_gradient2,
				      positive_gradient2);
    
    min_negative_gradient3 = std::min(min_negative_gradient3,
				      negative_gradient3);
    min_positive_gradient3 = std::max(min_positive_gradient3,
				      positive_gradient3);
  }

  /** @brief Tries to prune the given nodes using Monte Carlo
   *         sampling.
   *
   *         I am assuming this function gets called immediately after
   *         the mkernel_.Eval function.
   */
  bool MonteCarloEval(const Matrix &data, ArrayList<index_t> &indices,
		      ArrayList<TTree *> &nodes,
		      double relative_error, double threshold,
		      int num_samples, double total_n_minus_one_num_tuples, 
		      double num_tuples, double required_probability) {

    // Compute the number of (n - 1) tuples and leave-one-out
    // quantities.
    double num_jk_pairs, num_ik_pairs, num_ij_pairs;
    double leave_one_out_node_j_count_for_node_i,
      leave_one_out_node_k_count_for_node_i,
      leave_one_out_node_i_count_for_node_j,
      leave_one_out_node_k_count_for_node_j,
      leave_one_out_node_i_count_for_node_k,
      leave_one_out_node_j_count_for_node_k;
    double negative_gradient1_error, positive_gradient1_error,
      negative_gradient2_error, positive_gradient2_error,
      negative_gradient3_error, positive_gradient3_error;

    ComputeNumTwoTuples_(nodes, leave_one_out_node_j_count_for_node_i,
			 leave_one_out_node_k_count_for_node_i,
			 leave_one_out_node_i_count_for_node_j,
			 leave_one_out_node_k_count_for_node_j,
			 leave_one_out_node_i_count_for_node_k,
			 leave_one_out_node_j_count_for_node_k,
			 num_jk_pairs, num_ik_pairs, num_ij_pairs);
  
    // boolean flag for stating whether the three nodes are prunable,
    // and whether we should try pruning.
    bool prunable = false;

    // Temporary variables used for computation...
    double negative_gradient1, positive_gradient1, negative_gradient2,
      positive_gradient2, negative_gradient3, positive_gradient3;

    // Currently running order statistics and the raw sum and the
    // squared sums..
    double min_negative_gradient1 = 0, max_negative_gradient1 = -DBL_MAX;
    double min_positive_gradient1 = DBL_MAX, max_positive_gradient1 = 0;
    double min_negative_gradient2 = 0, max_negative_gradient2 = -DBL_MAX;
    double min_positive_gradient2 = DBL_MAX, max_positive_gradient2 = 0;
    double min_negative_gradient3 = 0, max_negative_gradient3 = -DBL_MAX;
    double min_positive_gradient3 = DBL_MAX, max_positive_gradient3 = 0;

    // Evaluate the three components required for force vector using
    // the pairwise squared distances among the three nodes.
    EvalGradients(distmat_, min_negative_gradient1, &max_negative_gradient1,
		  min_positive_gradient1, &max_positive_gradient1,
		  min_negative_gradient2, &max_negative_gradient2,
		  min_positive_gradient2, &max_positive_gradient2,
		  min_negative_gradient3, &max_negative_gradient3,
		  min_positive_gradient3, &max_positive_gradient3);

    // Sample a random 3-tuple from the i-th node, the j-th node and
    // the k-th node.
    for(index_t current_num_samples = 0; current_num_samples <
	num_samples; current_num_samples++) {

      // Select a valid 3-tuple sample.
      do {
	indices[0] = math::RandInt(nodes[0]->begin(), nodes[0]->end());
	indices[1] = math::RandInt(nodes[1]->begin(), nodes[1]->end());
	indices[2] = math::RandInt(nodes[2]->begin(), nodes[2]->end());
      } while(!(indices[0] < indices[1] && indices[1] < indices[2]));
      
      // Compute the pairwise distances among three particles to
      // complete the distance tables.
      EvalMinMaxSquaredDistances(data, indices);
      
      // Evaluate the three components required for force vector for
      // the current particle.
      EvalGradients(distmat_, negative_gradient1, NULL,
		    positive_gradient1, NULL, negative_gradient2, NULL, 
		    positive_gradient2, NULL, negative_gradient3, NULL, 
		    positive_gradient3, NULL);

      // Update the current statistics for all three components.
      UpdateStatistics_
	(negative_gradient1, positive_gradient1,
	 negative_gradient2, positive_gradient2,
	 negative_gradient3, positive_gradient3,
	 max_negative_gradient1, min_positive_gradient1, 
	 max_negative_gradient2, min_positive_gradient2, 
	 max_negative_gradient3, min_positive_gradient3);
      
      // Compute the current error and see if the error is satisifed and
      // recompute how many more to compute.      
      negative_gradient1_error = 0.5 * (max_negative_gradient1 - 
					min_negative_gradient1);
      positive_gradient1_error = 0.5 * (max_positive_gradient1 - 
					min_positive_gradient1);
      negative_gradient2_error = 0.5 * (max_negative_gradient2 - 
					min_negative_gradient2);
      positive_gradient2_error = 0.5 * (max_positive_gradient2 - 
					min_positive_gradient2);
      negative_gradient3_error = 0.5 * (max_negative_gradient3 - 
					min_negative_gradient3);
      positive_gradient3_error = 0.5 * (max_positive_gradient3 - 
					min_positive_gradient3);
      
      double node_i_additional_negative_gradient1_u,
	node_i_additional_positive_gradient1_l,
	node_i_additional_l1_norm_negative_gradient2_u,
	node_i_additional_l1_norm_positive_gradient2_l,
	node_j_additional_negative_gradient1_u,
	node_j_additional_positive_gradient1_l,
	node_j_additional_l1_norm_negative_gradient2_u,
	node_j_additional_l1_norm_positive_gradient2_l,
	node_k_additional_negative_gradient1_u,
	node_k_additional_positive_gradient1_l,
	node_k_additional_l1_norm_negative_gradient2_u,
	node_k_additional_l1_norm_positive_gradient2_l;
      
      // Additional bound changes due to sampling...
      ComputeAdditionalBoundChanges_
	(nodes, max_negative_gradient1, min_positive_gradient1,
	 max_negative_gradient2, min_positive_gradient2,
	 max_negative_gradient3, min_positive_gradient3,
	 leave_one_out_node_j_count_for_node_i,
	 leave_one_out_node_k_count_for_node_i,
	 leave_one_out_node_i_count_for_node_j,
	 leave_one_out_node_k_count_for_node_j,
	 leave_one_out_node_i_count_for_node_k,
	 leave_one_out_node_j_count_for_node_k,
	 num_jk_pairs, num_ik_pairs, num_ij_pairs,
	 node_i_additional_negative_gradient1_u,
	 node_i_additional_positive_gradient1_l,
	 node_i_additional_l1_norm_negative_gradient2_u,
	 node_i_additional_l1_norm_positive_gradient2_l,
	 node_j_additional_negative_gradient1_u,
	 node_j_additional_positive_gradient1_l,
	 node_j_additional_l1_norm_negative_gradient2_u,
	 node_j_additional_l1_norm_positive_gradient2_l,
	 node_k_additional_negative_gradient1_u,
	 node_k_additional_positive_gradient1_l,
	 node_k_additional_l1_norm_negative_gradient2_u,
	 node_k_additional_l1_norm_positive_gradient2_l);
      
      // If not prunable, recompute the required number of samples
      // to try again.
      prunable = Prunable_
	(nodes, negative_gradient1_error, positive_gradient1_error, 
	 negative_gradient2_error, positive_gradient2_error, 
	 negative_gradient3_error, positive_gradient3_error, 
	 node_i_additional_negative_gradient1_u,
	 node_i_additional_positive_gradient1_l,
	 node_i_additional_l1_norm_negative_gradient2_u,
	 node_i_additional_l1_norm_positive_gradient2_l,
	 node_j_additional_negative_gradient1_u,
	 node_j_additional_positive_gradient1_l,
	 node_j_additional_l1_norm_negative_gradient2_u,
	 node_j_additional_l1_norm_positive_gradient2_l,
	 node_k_additional_negative_gradient1_u,
	 node_k_additional_positive_gradient1_l,
	 node_k_additional_l1_norm_negative_gradient2_u,
	 node_k_additional_l1_norm_positive_gradient2_l,
	 num_jk_pairs, num_ik_pairs, num_ij_pairs, relative_error, threshold,
	 total_n_minus_one_num_tuples);

      if(prunable) {
	break;
      }
    }

    // If the three node tuple was prunable, then prune.
    if(prunable) {
      Prune_(nodes, negative_gradient1_error, positive_gradient1_error, 
	     negative_gradient2_error, positive_gradient2_error, 
	     negative_gradient3_error, positive_gradient3_error, 
	     min_negative_gradient1, max_negative_gradient1,
	     min_positive_gradient1, max_positive_gradient1,
	     min_negative_gradient2, max_negative_gradient2,
	     min_positive_gradient2, max_positive_gradient2,
	     min_negative_gradient3, max_negative_gradient3,
	     min_positive_gradient3, max_positive_gradient3, num_jk_pairs,
	     num_ik_pairs, num_ij_pairs);      
    }

    return prunable;
  }


  /** @brief Vanilla finite-difference component-wise relative error
   *  pruning.
   */
  bool Eval(const Matrix &data, ArrayList<index_t> &indices,
	    ArrayList<TTree *> &tree_nodes, double relative_error, 
	    double threshold, 
	    double total_n_minus_one_num_tuples, double num_tuples) {

    // First, compute the pairwise distance among the three nodes.
    EvalMinMaxSquaredDistances(tree_nodes);

    // The result of pruning
    bool prunable = false;

    // Temporary variables.
    double min_negative_gradient1, max_negative_gradient1, 
      min_positive_gradient1, max_positive_gradient1, min_negative_gradient2, 
      max_negative_gradient2, min_positive_gradient2, max_positive_gradient2,
      min_negative_gradient3, max_negative_gradient3, min_positive_gradient3, 
      max_positive_gradient3;

    // Then, evaluate the gradients.
    EvalGradients(distmat_, min_negative_gradient1, &max_negative_gradient1, 
		  min_positive_gradient1, &max_positive_gradient1, 
		  min_negative_gradient2, &max_negative_gradient2, 
		  min_positive_gradient2, &max_positive_gradient2, 
		  min_negative_gradient3, &max_negative_gradient3, 
		  min_positive_gradient3, &max_positive_gradient3);

    // If any of the components is computed to be NaN's or Inf's, then
    // do not attempt to approximate.
    if(isnan(min_negative_gradient1) || isnan(max_negative_gradient1) ||
       isnan(min_positive_gradient1) || isnan(max_positive_gradient1) ||
       isnan(min_negative_gradient2) || isnan(max_negative_gradient2) ||
       isnan(min_positive_gradient2) || isnan(max_positive_gradient2) ||
       isnan(min_negative_gradient3) || isnan(max_negative_gradient3) ||
       isnan(min_positive_gradient3) || isnan(max_positive_gradient3) ||
       isinf(min_negative_gradient1) || isinf(max_negative_gradient1) ||
       isinf(min_positive_gradient1) || isinf(max_positive_gradient1) ||
       isinf(min_negative_gradient2) || isinf(max_negative_gradient2) ||
       isinf(min_positive_gradient2) || isinf(max_positive_gradient2) ||
       isinf(min_negative_gradient3) || isinf(max_negative_gradient3) ||
       isinf(min_positive_gradient3) || isinf(max_positive_gradient3)) {
      return false;
    }
    
    // Compute approximation error.
    double negative_gradient1_error, positive_gradient1_error,
      negative_gradient2_error, positive_gradient2_error,
      negative_gradient3_error, positive_gradient3_error;
    ComputeGradientComponentError_
      (min_negative_gradient1, max_negative_gradient1,
       min_positive_gradient1, max_positive_gradient1,
       min_negative_gradient2, max_negative_gradient2,
       min_positive_gradient2, max_positive_gradient2,
       min_negative_gradient3, max_negative_gradient3,
       min_positive_gradient3, max_positive_gradient3,
       negative_gradient1_error, positive_gradient1_error,
       negative_gradient2_error, positive_gradient2_error,
       negative_gradient3_error, positive_gradient3_error);

    // Now determine whether all three nodes satisfy the pruning
    // conditions.
    double num_ik_pairs, num_ij_pairs, num_jk_pairs;
    double leave_one_out_node_j_count_for_node_i,
      leave_one_out_node_k_count_for_node_i,
      leave_one_out_node_i_count_for_node_j,
      leave_one_out_node_k_count_for_node_j,
      leave_one_out_node_i_count_for_node_k,
      leave_one_out_node_j_count_for_node_k;
    ComputeNumTwoTuples_(tree_nodes, leave_one_out_node_j_count_for_node_i,
			 leave_one_out_node_k_count_for_node_i,
			 leave_one_out_node_i_count_for_node_j,
			 leave_one_out_node_k_count_for_node_j,
			 leave_one_out_node_i_count_for_node_k,
			 leave_one_out_node_j_count_for_node_k,
			 num_jk_pairs, num_ik_pairs, num_ij_pairs);
    
    double node_i_additional_negative_gradient1_u,
      node_i_additional_positive_gradient1_l,
      node_i_additional_l1_norm_negative_gradient2_u,
      node_i_additional_l1_norm_positive_gradient2_l,
      node_j_additional_negative_gradient1_u,
      node_j_additional_positive_gradient1_l,
      node_j_additional_l1_norm_negative_gradient2_u,
      node_j_additional_l1_norm_positive_gradient2_l,
      node_k_additional_negative_gradient1_u,
      node_k_additional_positive_gradient1_l,
      node_k_additional_l1_norm_negative_gradient2_u,
      node_k_additional_l1_norm_positive_gradient2_l;
    
    // Additional bound changes due to sampling...
    ComputeAdditionalBoundChanges_
      (tree_nodes, max_negative_gradient1, min_positive_gradient1,
       max_negative_gradient2, min_positive_gradient2,
       max_negative_gradient3, min_positive_gradient3,
       leave_one_out_node_j_count_for_node_i,
       leave_one_out_node_k_count_for_node_i,
       leave_one_out_node_i_count_for_node_j,
       leave_one_out_node_k_count_for_node_j,
       leave_one_out_node_i_count_for_node_k,
       leave_one_out_node_j_count_for_node_k,
       num_jk_pairs, num_ik_pairs, num_ij_pairs,
       node_i_additional_negative_gradient1_u,
       node_i_additional_positive_gradient1_l,
       node_i_additional_l1_norm_negative_gradient2_u,
       node_i_additional_l1_norm_positive_gradient2_l,
       node_j_additional_negative_gradient1_u,
       node_j_additional_positive_gradient1_l,
       node_j_additional_l1_norm_negative_gradient2_u,
       node_j_additional_l1_norm_positive_gradient2_l,
       node_k_additional_negative_gradient1_u,
       node_k_additional_positive_gradient1_l,
       node_k_additional_l1_norm_negative_gradient2_u,
       node_k_additional_l1_norm_positive_gradient2_l);

    prunable = Prunable_(tree_nodes, negative_gradient1_error, 
			 positive_gradient1_error, negative_gradient2_error,
			 positive_gradient2_error, negative_gradient3_error,
			 positive_gradient3_error, 
			 node_i_additional_negative_gradient1_u,
			 node_i_additional_positive_gradient1_l,
			 node_i_additional_l1_norm_negative_gradient2_u,
			 node_i_additional_l1_norm_positive_gradient2_l,
			 node_j_additional_negative_gradient1_u,
			 node_j_additional_positive_gradient1_l,
			 node_j_additional_l1_norm_negative_gradient2_u,
			 node_j_additional_l1_norm_positive_gradient2_l,
			 node_k_additional_negative_gradient1_u,
			 node_k_additional_positive_gradient1_l,
			 node_k_additional_l1_norm_negative_gradient2_u,
			 node_k_additional_l1_norm_positive_gradient2_l,
			 num_jk_pairs, num_ik_pairs,
			 num_ij_pairs, relative_error, threshold,
			 total_n_minus_one_num_tuples);
    
    // Prune only if all three nodes can be approximated.
    if(prunable) {
      Prune_(tree_nodes, negative_gradient1_error, positive_gradient1_error, 
	     negative_gradient2_error, positive_gradient2_error, 
	     negative_gradient3_error, positive_gradient3_error,  
	     min_negative_gradient1, max_negative_gradient1, 
	     min_positive_gradient1, max_positive_gradient1, 
	     min_negative_gradient2, max_negative_gradient2, 
	     min_positive_gradient2, max_positive_gradient2, 
	     min_negative_gradient3, max_negative_gradient3, 
	     min_positive_gradient3, max_positive_gradient3,    
	     num_jk_pairs, num_ik_pairs, num_ij_pairs);
    }
    return prunable;
  }

};

#endif
