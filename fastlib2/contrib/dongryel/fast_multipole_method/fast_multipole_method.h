/** @file fast_multipole_method.h
 *
 *  This file contains an implementation of fast multipole method for
 *  the general $\frac{1}{r^{\lambda}}$ potential function. It uses an
 *  adpative octree implementation.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see fast_multipole_method_main.cc
 *  @bug No known bugs.
 */

#ifndef FAST_MULTIPOLE_METHOD_H
#define FAST_MULTIPOLE_METHOD_H

#include "fastlib/fastlib.h"
#include "fmm_stat.h"
#include "mlpack/kde/dualtree_kde_common.h"
#include "mlpack/series_expansion/inverse_pow_dist_farfield_expansion.h"
#include "mlpack/series_expansion/inverse_pow_dist_local_expansion.h"
#include "contrib/dongryel/proximity_project/gen_hypercube_tree.h"

class FastMultipoleMethod {

 private:

  ////////// Private Member Variables //////////

  double lambda_;

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module_;

  /** @brief The boolean flag to control the leave-one-out computation.
   */
  bool leave_one_out_;

  /** @brief The series expansino auxilary object.
   */
  InversePowDistSeriesExpansionAux sea_;

  /** @brief The shuffled query particle set.
   */
  Matrix shuffled_query_particle_set_;

  /** @brief The shuffled reference particle set.
   */
  Matrix shuffled_reference_particle_set_;

  /** @brief The shuffled reference particle charge set.
   */
  Vector shuffled_reference_particle_charge_set_;

  /** @brief The octree containing the entire particle set.
   */
  proximity::GenHypercubeTree<FmmStat> *tree_;

  /** @brief The list of nodes on each level.
   */
  ArrayList< ArrayList <proximity::GenHypercubeTree<FmmStat> *> > nodes_in_each_level_;

  /** @brief The number of query particles in the particle set.
   */
  int num_query_particles_;
  
  /** @brief The number of reference particles in the particle set.
   */
  int num_reference_particles_;

  /** @brief The permutation mapping indices of the particle indices
   *         to original order.
   */
  ArrayList< ArrayList<index_t> > old_from_new_index_;

  /** @brief The permutation mapping indices of the shuffled indices
   *         from the original order.
   */
  ArrayList< ArrayList<index_t> > new_from_old_index_;

  ////////// Private Member Functions //////////
  void FormMultipoleExpansions_() {
    
    Vector node_center;
    node_center.Init(shuffled_reference_particle_set_.n_rows());

    // Start from the most bottom level, and work your way up.
    for(index_t level = nodes_in_each_level_.size() - 1; level >= 0; level--) {

      // The references to the nodes on the current level.
      ArrayList<proximity::GenHypercubeTree<FmmStat> *> 
	&nodes_on_current_level = nodes_in_each_level_[level];

      // Iterate over each node in the list.
      for(index_t n = 0; n < nodes_on_current_level.size(); n++) {
	
	proximity::GenHypercubeTree<FmmStat> *node = nodes_on_current_level[n];
	
	// Compute the node center.
	for(index_t i = 0; i < shuffled_reference_particle_set_.n_rows(); 
	    i++) {
	  node_center[i] = 0.5 * (node->bound().get(i).lo +
				  node->bound().get(i).hi);
	}

	// Initialize the far-field expansion of the current node.
	node->stat().farfield_expansion_.Init(node_center, &sea_);

	// If the current node is a leaf node, then compute
	// exhaustively its far-field moments.
	if(node->is_leaf()) {
	  node->stat().farfield_expansion_.AccumulateCoeffs
	    (shuffled_reference_particle_set_,
	     shuffled_reference_particle_charge_set_,
	     node->begin(0), node->end(0), sea_.get_max_order());
	}
	
	// Otherwise, translate the moments owned by the children in a
	// bottom-up fashion.
	else {
	  for(index_t child = 0; child < node->num_children(); child++) {
	    node->stat().farfield_expansion_.TranslateFromFarField
	      (node->get_child(child)->stat().farfield_expansion_);
	  }
	}
      }
    }
  }

 public:

  void Compute() {
    
    // Upward pass: Form multipole expansions.
    FormMultipoleExpansions_();

    // Downward pass
    
  }

  void Init(const Matrix &queries, const Matrix &references,
	    const Matrix &rset_weights, bool queries_equal_references, 
	    struct datanode *module_in) {
    
    // Point to the incoming module.
    module_ = module_in;
    
    // Set the flag for whether to perform leave-one-out computation.
    leave_one_out_ = fx_param_exists(module_in, "loo") &&
      (queries.ptr() == references.ptr());

    // Read in the number of points owned by a leaf.
    int leaflen = std::max((long long int) 3, 
			   fx_param_int(module_in, "leaflen", 20));

    // Set the number of query particles and reference particles
    // accordingly.
    num_query_particles_ = queries.n_cols();
    num_reference_particles_ = references.n_cols();
    
    // Approporiately initialize the query/reference sets.
    ArrayList<Matrix *> particle_sets;
    particle_sets.Init();
    shuffled_reference_particle_set_.Copy(references);
    particle_sets.PushBackCopy(&shuffled_reference_particle_set_);

    if(queries.ptr() != references.ptr()) {
      shuffled_query_particle_set_.Copy(queries);
      particle_sets.PushBackCopy(&shuffled_query_particle_set_);
    }
    else {
      shuffled_query_particle_set_.Alias(shuffled_reference_particle_set_);
    }

    // Copy over the reference charge set.
    shuffled_reference_particle_charge_set_.Init(rset_weights.n_cols());
    for(index_t i = 0; i < rset_weights.n_cols(); i++) {
      shuffled_reference_particle_charge_set_[i] = rset_weights.get(0, i);
    }

    printf("Before permuting the reference set...\n");
    shuffled_reference_particle_set_.PrintDebug();
    printf("Before permuting the query set...\n");
    shuffled_query_particle_set_.PrintDebug();

    // Construct query and reference trees. Shuffle the reference
    // weights according to the permutation of the reference set in
    // the reference tree.
    fx_timer_start(NULL, "tree_d");
    tree_ = proximity::MakeGenHypercubeTree(particle_sets, leaflen,
					    &nodes_in_each_level_,
					    &old_from_new_index_,
					    &new_from_old_index_);
    fx_timer_stop(NULL, "tree_d");
    
    printf("After permuting...\n");
    shuffled_reference_particle_set_.PrintDebug();
    printf("After permuting the query set...\n");
    shuffled_query_particle_set_.PrintDebug();

    for(index_t i = 0; i < old_from_new_index_.size(); i++) {
      for(index_t j = 0; j < old_from_new_index_[i].size(); j++) {
	printf("%d ", old_from_new_index_[i][j]);
      }
      printf("\n");
    }

    // Shuffle the reference particle charges according to the
    // permutation of the reference particle set.
    DualtreeKdeCommon::ShuffleAccordingToPermutation
      (shuffled_reference_particle_charge_set_, old_from_new_index_[0]);

    // Retrieve the lambda order needed for expansion.
    lambda_ = fx_param_double(module_, "lambda", 1.0);

    // Initialize the series expansion auxliary object.
    sea_.Init(lambda_, 8, references.n_rows());
  }
};

#endif
