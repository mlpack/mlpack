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
#include "contrib/dongryel/proximity_project/gen_hypercube_tree.h"

class FastMultipoleMethod {

 private:

  ////////// Private Member Variables //////////

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module_;

  /** @brief The boolean flag to control the leave-one-out computation.
   */
  bool leave_one_out_;

  /** @brief The combined particle set.
   */
  Matrix particle_set_;

  /** @brief The octree containing the entire particle set.
   */
  proximity::GenHypercubeTree tree_;

  /** @brief The number of query particles in the particle set.
   */
  int num_query_particles_;
  
  /** @brief The number of reference particles in the particle set.
   */
  int num_reference_particles_;

 public:
  
  void Init(const Matrix &queries, const Matrix &references,
	    const Matrix &rset_weights, bool queries_equal_references, 
	    struct datanode *module_in) {
    
    // point to the incoming module
    module_ = module_in;
    
    // Set the flag for whether to perform leave-one-out computation.
    leave_one_out_ = fx_param_exists(module_in, "loo") &&
      (queries.ptr() == references.ptr());

    // Read in the number of points owned by a leaf.
    int leaflen = fx_param_int(module_in, "leaflen", 20);

    // Set the number of query particles and reference particles
    // accordingly.
    num_query_particles_ = queries.n_cols();
    num_reference_particles_ = references.n_cols();

    // Combine the two sets to form the global particle set.
    particle_set_.Init(queries.n_rows(), num_query_particles_ +
		       num_reference_particles_);
    for(index_t q = 0; q < queries.n_cols(); q++) {
      Vector query_column_destination;
      particle_set_.MakeColumnVector(q, &query_column_destination);
      query_column_destination.CopyValues(queries.GetColumnPtr(q));
    }
    for(index_t r = 0; r < references.n_cols(); r++) {
      Vector reference_column_destination;
      particle_set_.MakeColumnVector(queries.n_cols() + r,
				     &reference_column_destination);
      reference_column_destination.CopyValues(references.GetColumnPtr(r));
    }

    // Construct query and reference trees. Shuffle the reference
    // weights according to the permutation of the reference set in
    // the reference tree.
    fx_timer_start(NULL, "tree_d");

    fx_timer_stop(NULL, "tree_d");    
  }
};

#endif
