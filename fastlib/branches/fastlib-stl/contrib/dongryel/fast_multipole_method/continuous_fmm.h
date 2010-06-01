/** @file continuous_fmm.h
 *
 *  This file contains an implementation of the continuous fast
 *  multipole method.
 *
 *  article{white1994cfm,
 *    title={{The continuous fast multipole method}},
 *    author={White, C.A. and Johnson, B.G. and Gill, P.M.W. and 
 *            Head-Gordon, M.},
 *    journal={Chemical Physics Letters (ISSN 0009-2614)},
 *    volume={230},
 *    number={1-2},
 *    year={1994}
 *  }
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @see continuous_fmm_main.cc
 *  @bug No known bugs.
 */

#ifndef CONTINUOUS_FMM_H
#define CONTINUOUS_FMM_H

#include "fastlib/fastlib.h"
#include "fmm_stat.h"
#include "mlpack/kde/inverse_normal_cdf.h"
#include "mlpack/series_expansion/inverse_pow_dist_kernel.h"
#include "mlpack/series_expansion/inverse_pow_dist_farfield_expansion.h"
#include "mlpack/series_expansion/inverse_pow_dist_local_expansion.h"
#include "contrib/dongryel/proximity_project/cfmm_tree.h"
#include "contrib/dongryel/multitree_template/multitree_utility.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/march/fock_matrix/fock_impl/shell_pair.h"
#include "contrib/march/libint/include/libint/libint.h"
#include "contrib/march/libint/include/libint/hrr_header.h"
#include "contrib/march/libint/include/libint/vrr_header.h"
#include "contrib/march/fock_matrix/fock_impl/integral_tensor.h"


const fx_entry_doc cfmm_fmm_mod_entries[] = {
{"charge_thresh", FX_PARAM, FX_DOUBLE, NULL,
  "The screening threshold for including a charge distribution.\n"},
  {"leaflen", FX_PARAM, FX_INT, NULL, 
  "The number of points owned by a leaf.  Default: 1\n"},
  {"precision", FX_PARAM, FX_DOUBLE, NULL,
  "epsilon used in the definition of extent of distributions.  Default: 0.1\n"},
  {"min_ws_index", FX_PARAM, FX_INT, NULL, 
  "The smallest possible well-separated index at which distributions will be\n"
  "approximated with multipoles.  Default: 2\n"},
  {"max_tree_depth", FX_PARAM, FX_INT, NULL, 
   "FILL ME IN!\n"},
  {"num_exact_computations", FX_RESULT, FX_INT, NULL, 
    "The number of interactions computed exactly in the base case.\n"},
  {"num_approx_computations", FX_RESULT, FX_INT, NULL, 
    "The number of interactions approximated with multipole expansions.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc cfmm_fmm_mod_doc = {
  cfmm_fmm_mod_entries, NULL,
  "Algorithm module for actual multipole code in CFMM.\n"
};


class ContinuousFmm {

 private:

  double sqrt_pi_;

  ////////// Private Member Variables //////////

  double lambda_;

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module_;

  /** @brief The boolean flag to control the leave-one-out computation.
   */
  bool leave_one_out_;

  /** @brief The inverse distance kernel object.
   */
  InversePowDistKernel kernel_;

  /** @brief The series expansion auxilary object.
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

  /** @brief The shuffled reference particle bandwidth set.
   */
  Vector shuffled_reference_particle_bandwidth_set_;

  /** @brief The shuffled reference particle extent number.
   */
  Vector shuffled_reference_particle_extent_set_;

  /** @brief The octree containing the entire particle set.
   */
  proximity::CFmmTree<FmmStat> *tree_;

  /** @brief The list of nodes on each level.
   */
  ArrayList< ArrayList <proximity::CFmmTree<FmmStat> *> > nodes_in_each_level_;

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

  /** @brief The accumulated potential for each query particle.
   */
  Vector potentials_;
  
  /** @brief The number of potentials computed in the base case.
   */
  int num_exact_computations_;
  
  /**
   * @brief Stores the results of the near-field computations for use with 
   * Libint, used pointer to avoid copying - added by BM, 8/18/09
   */
  Matrix* near_field_results_;
  
  /**
   * @brief The list of shell pairs, needed to use the libint wrappers in the 
   * base case - added by BM, 8/18/09
   */
  ArrayList<ShellPair*> shell_pairs_;
  
  /**
   * @brief The density matrix, needed to accurately compute the base case
   * used pointer to avoid copying - added by BM, 8/18/09
   */
  Matrix* density_;
  
  ////////// Private Member Functions //////////

  void ReshuffleResults_(Vector &to_be_reshuffled) {

    index_t query_point_indexing = 
      (shuffled_reference_particle_set_.ptr() == 
       shuffled_query_particle_set_.ptr()) ? 0:1;

    // Reshuffle the results to account for dataset reshuffling
    // resulted from tree constructions.
    Vector tmp_results;
    tmp_results.Init(to_be_reshuffled.length());
    
    for(index_t i = 0; i < tmp_results.length(); i++) {
      tmp_results[old_from_new_index_[query_point_indexing][i]] =
	to_be_reshuffled[i];
    }
    for(index_t i = 0; i < tmp_results.length(); i++) {
      to_be_reshuffled[i] = tmp_results[i];
    }
  }
  
  // Lame way to do this - added by BM, 8/18/09
  void ShuffleShellPairList(ArrayList<ShellPair*>& pairs, 
                            const ArrayList<index_t>& permutation) {
    
    ArrayList<ShellPair*> temp;
    temp.Init(pairs.size());
    
    for (int i = 0; i < pairs.size(); i++) {
      temp[i] = pairs[permutation[i]];
    }
    pairs.Swap(&temp);
    
  } // ShuffleShellPairsList

  void FormMultipoleExpansions_() {
    
    Vector node_center;
    node_center.Init(shuffled_reference_particle_set_.n_rows());

    // Start from the most bottom level, and work your way up to the
    // direct children of the root node.
    for(index_t level = nodes_in_each_level_.size() - 1; level >= 0; level--) {

      // The references to the nodes on the current level.
      ArrayList<proximity::CFmmTree<FmmStat> *> 
	&nodes_on_current_level = nodes_in_each_level_[level];

      // Iterate over each node in the list.
      for(index_t n = 0; n < nodes_on_current_level.size(); n++) {
	
	proximity::CFmmTree<FmmStat> *node = nodes_on_current_level[n];
	
	// Compute the node center.
	for(index_t i = 0; i < shuffled_reference_particle_set_.n_rows(); 
	    i++) {
	  node_center[i] = node->bound().get(i).mid();
	}

	// Initialize the far-field expansion of the current node.
	node->stat().farfield_expansion_.Init(node_center, &sea_);
	node->init_flag_ = true;

	// Also initialize the local expansion of the current node (to
	// be used in the downward pass later).
	node->stat().local_expansion_.Init(node_center, &sea_);

	// If the current node is a leaf node, then compute
	// exhaustively its far-field moments.
	if(node->is_leaf()) {
	  node->stat().farfield_expansion_.AccumulateCoeffs
	    (shuffled_reference_particle_set_,
	     shuffled_reference_particle_charge_set_,
	     node->begin(0), node->end(0), sea_.get_max_order());
	}
	
	// Otherwise, translate the moments owned by the partitions...
	else {
	  for(index_t p = 0; p < node->partitions_based_on_ws_indices_.size();
	      p++) {

	    node->stat().farfield_expansion_.TranslateFromFarField
	      (node->partitions_based_on_ws_indices_[p]->stat_.
	       farfield_expansion_);
	  }
	}

	// If the current node has a "ws-node" parent, then add the
	// contribution to it... Of course, we need to initialize the
	// moments set before adding...
	if(node->parent_ != NULL && (!(node->parent_->init_flag_))) {

	  // The node center is the node that owns the partition, so
	  // it's the parent's parent...
	  for(index_t i = 0; i < shuffled_reference_particle_set_.n_rows();
	      i++) {
	    node_center[i] = (node->parent_->parent_->bound()).get(i).mid();
	  }

	  node->parent_->stat_.farfield_expansion_.Init(node_center, &sea_);
	  node->parent_->stat_.local_expansion_.Init(node_center, &sea_);
	  node->parent_->init_flag_ = true;
	}
	if(node->parent_ != NULL) {
	  node->parent_->stat_.farfield_expansion_.TranslateFromFarField
	    (node->stat().farfield_expansion_);
	}
	
      } // iterating over each node on the current level...
    } // iterating over each level set...
  }

  void EvaluateMultipoleExpansion_
  (proximity::CFmmTree<FmmStat> *query_node, 
   proximity::CFmmTree<FmmStat> *reference_node) {

    index_t query_point_indexing = 
      (shuffled_reference_particle_set_.ptr() == 
       shuffled_query_particle_set_.ptr()) ? 0:1;

    for(index_t q = query_node->begin(query_point_indexing); 
	q < query_node->end(query_point_indexing); q++) {
      
      potentials_[q] += 
	reference_node->stat().farfield_expansion_.EvaluateField
	(shuffled_query_particle_set_, q, sea_.get_max_order());
    }
  }

  void BaseCase_(proximity::CFmmTree<FmmStat> *query_node,
                 proximity::CFmmTree<FmmStat> *reference_node,
                 Vector &potentials) {
    
    
    index_t query_point_indexing = 
    (shuffled_reference_particle_set_.ptr() == 
     shuffled_query_particle_set_.ptr()) ? 0:1;
    
    
    for(index_t q = query_node->begin(query_point_indexing); 
        q < query_node->end(query_point_indexing); q++) {
      
      // q and r should be able to index into shell_pairs_ as well
    
      // Get the query point.
      //const double *q_col = shuffled_query_particle_set_.GetColumnPtr(q);
      ShellPair* query = shell_pairs_[q];
      
      Matrix query_results;
      query_results.Init(query->M_Shell()->num_functions(), 
                         query->N_Shell()->num_functions());
      query_results.SetZero();
      
      for(index_t r = reference_node->begin(0); r < reference_node->end(0);
          r++) {
        
        ShellPair* reference = shell_pairs_[r];
        
        IntegralTensor integrals;
        eri::ComputeShellIntegrals(*query, *reference, &integrals);
        
        // now contract with the density and sum into the results matrix
        integrals.ContractCoulomb(reference->M_Shell()->matrix_indices(),
                                  reference->N_Shell()->matrix_indices(),
                                  *density_, &query_results, 
                                  (reference->M_Shell() == reference->N_Shell()));
        
        /// old base case code
        /*
        // Compute the pairwise distance, if the query and the
        // reference are not the same particle.
        // We need to compute the self-interaction as well - BM
        if(leave_one_out_ && q == r) {
          continue;
        }
        const double *r_col = shuffled_reference_particle_set_.GetColumnPtr(r);
        
        double sq_dist = la::DistanceSqEuclidean
          (shuffled_query_particle_set_.n_rows(), q_col, r_col);
        double dist = sqrt(sq_dist);
        double erf_argument = 
          sqrt(shuffled_reference_particle_bandwidth_set_[q] *
               shuffled_reference_particle_bandwidth_set_[r] /
               (shuffled_reference_particle_bandwidth_set_[q] +
                shuffled_reference_particle_bandwidth_set_[r]));
        
        // This implements the kernel function used for the base case
        // in the page 2 of the CFMM paper...
        
        if(dist > 0) {
          
          potentials[q] += shuffled_reference_particle_charge_set_[r] * 
          erf(erf_argument * dist) / dist;
        }
        else {
          if (q == r) {
            potentials[q] += shuffled_reference_particle_charge_set_[r] * 
            erf_argument * 2 / sqrt_pi_;
          }
          else {
            // not sure if I should really multiply by 2 here
            potentials[q] += shuffled_reference_particle_charge_set_[r] * erf_argument
            * 2 / sqrt_pi_;
          }
        }
        */
        
      } // for r
      
      eri::AddSubmatrix(query->M_Shell()->matrix_indices(), 
                        query->N_Shell()->matrix_indices(),
                        query_results, near_field_results_);
      
      // handle entries below the diagonal
      if (query->M_Shell() != query->N_Shell()) {
        
        Matrix query_trans;
        la::TransposeInit(query_results, &query_trans);
        eri::AddSubmatrix(query->N_Shell()->matrix_indices(), 
                          query->M_Shell()->matrix_indices(),
                          query_trans, near_field_results_);
        
      }
      
    } // for q
    
    num_exact_computations_ += (query_node->end(query_point_indexing)
                                - query_node->begin(query_point_indexing))
                               * (reference_node->end(0)
                                  - reference_node->begin(0));
    
  }

  void EvaluateLocalExpansion_(proximity::CFmmTree<FmmStat> *query_node) {

    index_t query_point_indexing = 
      (shuffled_reference_particle_set_.ptr() == 
       shuffled_query_particle_set_.ptr()) ? 0:1;

    for(index_t q = query_node->begin(query_point_indexing); 
	q < query_node->end(query_point_indexing); q++) {

      // Evaluate the local expansion at the current query point.
      potentials_[q] += query_node->stat().local_expansion_.
	EvaluateField(shuffled_query_particle_set_, q,
		      sea_.get_max_order());
    }    
  }

  void TransmitLocalExpansionToChildren_
  (proximity::CFmmTree<FmmStat> *query_node) {
    
    // Two step process: first transmit the local expansion of the
    // current query node to each local expansion of the two
    // partitions, then for each partition, transmit to its children.

    for(index_t p = 0; p < query_node->partitions_based_on_ws_indices_.size();
	p++) {
      
      query_node->stat().local_expansion_.TranslateToLocal
	(query_node->partitions_based_on_ws_indices_[p]->stat_.
	 local_expansion_);

      for(index_t c = 0; c < query_node->partitions_based_on_ws_indices_[p]
	    ->num_children(); c++) {
	
	// Query child.
	proximity::CFmmTree<FmmStat> *query_child_node =
	  query_node->partitions_based_on_ws_indices_[p]->get_child(c);
	
	query_node->partitions_based_on_ws_indices_[p]->stat_.
	  local_expansion_.TranslateToLocal
	  (query_child_node->stat().local_expansion_);
      }
    }
  }

  void DownwardPass_() {

    index_t query_point_indexing = 
      (shuffled_reference_particle_set_.ptr() ==
       shuffled_query_particle_set_.ptr()) ? 0:1;

    // Start from the top level and descend down the tree.
    for(index_t level = 1; level < nodes_in_each_level_.size(); level++) {
      
      // Retrieve the nodes on the current level and the level above.
      const ArrayList<proximity::CFmmTree<FmmStat> * > 
	&nodes_on_current_level = nodes_in_each_level_[level];
      const ArrayList<proximity::CFmmTree<FmmStat> * >
	&nodes_on_previous_level = nodes_in_each_level_[level - 1];

      // Iterate over each query node in this level.
      for(index_t n = 0; n < nodes_on_current_level.size(); n++) {

	// The pointer to the current query node.
	proximity::CFmmTree<FmmStat> *node = nodes_on_current_level[n];

	// If the node does not contain any query points, then skip
	// it.
	if(node->count(query_point_indexing) == 0) {
	  continue;
	}

	// Get the parent node of the current query node.
	proximity::CFmmTree<FmmStat> *parent_node = node->parent_->parent_;
	
	// For each parent, get its nearest neighbors and
	// subsequently the children under it.
	for(index_t i = 0; i < nodes_on_previous_level.size(); i++) {
	  
	  proximity::CFmmTree<FmmStat> *possible_nn_of_parent =
	    nodes_on_previous_level[i];

	  double min_dist = sqrt(parent_node->bound().MinDistanceSq
				 (possible_nn_of_parent->bound()));
	  
	  // Consider the children under the nearest neighbor of
	  // parent.
	  if(min_dist == 0) {
	    for(index_t p = 0; p < possible_nn_of_parent->
		  partitions_based_on_ws_indices_.size(); p++) {
	      
	      proximity::CFmmWellSeparatedTree<FmmStat> *partition = 
		possible_nn_of_parent->partitions_based_on_ws_indices_[p];

	      for(index_t c = 0; c < partition->num_children(); c++) {
		
		// The current child.
		proximity::CFmmTree<FmmStat> *current_reference_child =
		  partition->get_child(c);

		// If the reference child contains no points, then
		// skip it.
		if(current_reference_child->count(0) == 0) {
		  continue;
		}

		// Compute the distance between the current query node
		// to the current reference child on the same level.
		double min_dist_between_query_and_reference =
		  sqrt(node->bound().MinDistanceSq
		       (current_reference_child->bound()));

		if(min_dist_between_query_and_reference > 0) {

		  // The required WS index for the query and the
		  // reference to be approximated using F2L
		  // translation.
		  index_t required_ws_index = -1;

		  // Test whether the query and the reference are
		  // under the same branch of CFMM tree.
		  if(node->parent_ == current_reference_child->parent_) {

		    required_ws_index = node->parent_->
		      well_separated_indices_[0];
		  }
		  else {
		    required_ws_index = 
		      (int) ceil(0.5 * (node->well_separated_indices_[0] +
					current_reference_child->
					well_separated_indices_[0]));
		  }

		  // If the two nodes are at least WS nodes apart,
		  // then translate far to local. Otherwise,
		  // accumulate exhaustively.
		  if(min_dist_between_query_and_reference >= 
		     required_ws_index * node->side_length()) {
		    current_reference_child->stat().farfield_expansion_.
		      TranslateToLocal(node->stat().local_expansion_,
				       sea_.get_max_order());
            //printf("multipole approx\n");
		  }
		  else {
		    BaseCase_(node, current_reference_child, potentials_);
		  }

		} // end of determining whether the query and the
		  // reference are at least zero apart.

		else if(node->is_leaf() &&
			node != current_reference_child) {
		  BaseCase_(node, current_reference_child, potentials_);
		}


	      } // end of looping over each child in the partition.
	    } // end of looping over each partition.
	  
	  } // end of determining whether the node is nearest neighbor
	    // of the parent of the current query node.
	} // end of looping over each node on the previous level.
	

	// If the node is a leaf node, then we need to compute
	// self-interaction as well, and evaluate the local expansion
	// formed inside it. Otherwise, translate the local moments
	// downwards.
	if(node->is_leaf()) {
	  EvaluateLocalExpansion_(node);
	  
          // If the node contains any reference points, then we have
          // to do the self-interactions among the node.
          if(node->count(0) > 0) {
	    BaseCase_(node, node, potentials_);
          }
	}

	// Otherwise, we need to pass it down.
	else {
	  TransmitLocalExpansionToChildren_(node);
	}

      } // end of iterating over each query box node on this level...
      
    } // end of iterating over each level...
  }

  void OutputResultsToFile_(const Vector &results, const char *fname) {

    FILE *stream = fopen(fname, "w+");
    for(index_t q = 0; q < results.length(); q++) {
      fprintf(stream, "%g\n", results[q]);
    }    
    fclose(stream);
  }

 public:

  ContinuousFmm() {
  }

  /*
  ~ContinuousFmm() {
    if(tree_ != NULL) {
      //tree_->~CFmmTree();
      delete tree_;
      tree_ = NULL;
    }
    
    old_from_new_index_.Clear();
    new_from_old_index_.Clear();
    nodes_in_each_level_.Clear();
    //sea_.~InversePowDistSeriesExpansionAux();
    //shuffled_reference_particle_set_.Destruct();
    //shuffled_reference_particle_charge_set_.Destruct();
  }
  */
  void Destruct() {
   
    if(tree_ != NULL) {
      //tree_->~CFmmTree();
      delete tree_;
      tree_ = NULL;
    }
    old_from_new_index_.Clear();
    new_from_old_index_.Clear();
    nodes_in_each_level_.Clear();
    shuffled_reference_particle_set_.Destruct();
    shuffled_query_particle_set_.Destruct();
    //sea_.~InversePowDistSeriesExpansionAux();
    
  }

  void NaiveCompute(Vector *naively_computed_potentials) {
    
    printf("Starting the naive computation...\n");

    naively_computed_potentials->Init(shuffled_query_particle_set_.n_cols());

    fx_timer_start(NULL, "naive_fmm_compute");

    // Call the base case...
    naively_computed_potentials->SetZero();
    BaseCase_(tree_, tree_, *naively_computed_potentials);

    fx_timer_stop(NULL, "naive_fmm_compute");

    printf("Finished the naive computation...\n");

    // Reshuffle the results according to the permutation.
    ReshuffleResults_(*naively_computed_potentials);

    // Output the results to the file.
    //OutputResultsToFile_(*naively_computed_potentials, "naive_fmm_output.txt");
    
  }
 
  /*
  void Update(const Matrix &rset_weights) {
   
    for(index_t i = 0; i < rset_weights.n_cols(); i++) {
      shuffled_reference_particle_charge_set_[i] = rset_weights.get(0, i);
    }
    
    MultiTreeUtility::ShuffleAccordingToPermutation
        (shuffled_reference_particle_charge_set_, old_from_new_index_[0]);
    
  
  }
*/
  
  void Compute(Vector* potentials_out) {
    
    printf("Starting the computation...\n");

    fx_timer_start(module_, "fmm_compute");

    // Reset the accumulated sum.
    potentials_.SetZero();

    // Upward pass: Form multipole expansions.
    FormMultipoleExpansions_();

    // Downward pass
    if(tree_->is_leaf()) {
      BaseCase_(tree_, tree_, potentials_);
    }
    else {
      DownwardPass_();
    }

    fx_timer_stop(module_, "fmm_compute");

    printf("Finished the computation...\n");

    // Reshuffle the results to account for dataset reshuffling
    // resulted from tree constructions.
    ReshuffleResults_(potentials_);
    
    // Output the results to the file.
    //OutputResultsToFile_(potentials_, "fast_fmm_output.txt");
    
    if (potentials_out != NULL) {
      potentials_out->Copy(potentials_);
    }
    
    fx_result_int(module_, "num_exact_computations", num_exact_computations_);
    fx_result_int(module_, "num_approx_computations", 
                  (num_query_particles_ * num_reference_particles_)
                     - num_exact_computations_);
    
  }
  
  void Reset(const Matrix& new_charges) {
    
    for(index_t i = 0; i < new_charges.n_cols(); i++) {
      shuffled_reference_particle_charge_set_[i] = new_charges.get(0, i);
    }
    
    MultiTreeUtility::ShuffleAccordingToPermutation
    (shuffled_reference_particle_charge_set_, old_from_new_index_[0]);
    
    potentials_.SetZero();
    
    // the remaining problems are: kernel_, sea_, and the tree
    
  }

  void Init(const Matrix &queries, const Matrix &references,
            const Matrix &rset_weights, const Matrix &rset_bandwidths,
            bool queries_equal_references, struct datanode *module_in, 
            Matrix* near_field, const ArrayList<ShellPair*>& shell_pairs, 
            Matrix* density) {
    
    // Point to the incoming module.
    module_ = module_in;
    
    // Set the flag for whether to perform leave-one-out computation.
    //leave_one_out_ = (queries.ptr() == references.ptr());
    // this should always be false, since the self interaction is important - BM
    leave_one_out_ = false;
    
    // Read in the number of points owned by a leaf.
    int leaflen = std::max((long long int) 1, 
			   fx_param_int(module_in, "leaflen", 1));

    // Set the number of query particles and reference particles
    // accordingly.
    num_query_particles_ = queries.n_cols();
    num_reference_particles_ = references.n_cols();
    
    // Approporiately initialize the query/reference sets.
    ArrayList<Matrix *> particle_sets;
    particle_sets.Init();
    shuffled_reference_particle_set_.Copy(references);
    *(particle_sets.PushBackRaw()) = &shuffled_reference_particle_set_;

    if(queries.ptr() != references.ptr()) {
      shuffled_query_particle_set_.Copy(queries);
      *(particle_sets.PushBackRaw()) = &shuffled_query_particle_set_;
    }
    else {
      shuffled_query_particle_set_.Alias(shuffled_reference_particle_set_);
    }

    // Copy over the reference charge set.
    shuffled_reference_particle_charge_set_.Init(rset_weights.n_cols());
    for(index_t i = 0; i < rset_weights.n_cols(); i++) {
      shuffled_reference_particle_charge_set_[i] = rset_weights.get(0, i);
    }

    // Copy over the reference bandwidth set and initialize the extent
    // for each particle.
    shuffled_reference_particle_bandwidth_set_.Init(rset_weights.n_cols());
    shuffled_reference_particle_extent_set_.Init(rset_weights.n_cols());
    for(index_t i = 0; i < rset_weights.n_cols(); i++) {
      shuffled_reference_particle_bandwidth_set_[i] = 
	rset_bandwidths.get(0, i);
      shuffled_reference_particle_extent_set_[i] = 
	sqrt(2.0 / shuffled_reference_particle_bandwidth_set_[i]) *
	InverseNormalCDF::Compute
	(1.0 - 0.5 * fx_param_double(module_, "precision", 0.1));
    }

    // Construct query and reference trees. Shuffle the reference
    // weights according to the permutation of the reference set in
    // the reference tree.
    ArrayList<Vector *> target_sets;
    target_sets.Init();
    *(target_sets.PushBackRaw()) = &shuffled_reference_particle_extent_set_;
    fx_timer_start(NULL, "tree_d");
    tree_ = proximity::MakeCFmmTree
      (particle_sets, target_sets, leaflen,
       fx_param_int(module_, "min_ws_index", 2),
       fx_param_int(module_, "max_tree_depth", 3),
       &nodes_in_each_level_, &old_from_new_index_, &new_from_old_index_);
    fx_timer_stop(NULL, "tree_d");

    printf("Constructed the tree...\n");
    //ot::Print(old_from_new_index_);
    //tree_->Print();
    
    // Shuffle the reference particle charges, the reference particle
    // bandwidths, and the reference particle extents according to the
    // permutation of the reference particle set.
    MultiTreeUtility::ShuffleAccordingToPermutation
      (shuffled_reference_particle_charge_set_, old_from_new_index_[0]);
    MultiTreeUtility::ShuffleAccordingToPermutation
      (shuffled_reference_particle_bandwidth_set_, old_from_new_index_[0]);
    MultiTreeUtility::ShuffleAccordingToPermutation
      (shuffled_reference_particle_extent_set_, old_from_new_index_[0]);

    //ot::Print(shuffled_reference_particle_charge_set_);

    // Retrieve the lambda order needed for expansion. The CFMM uses
    // the Coulombic kernel, hence always 1...
    lambda_ = 1.0;

    // Initialize the kernel.
    kernel_.Init(lambda_, queries.n_rows());

    // Initialize the series expansion auxliary object.
    sea_.Init(lambda_, fx_param_int(module_, "order", 2), references.n_rows());

    // Allocate the vector for storing the accumulated potential.
    potentials_.Init(shuffled_query_particle_set_.n_cols());

    // Compute PI.
    sqrt_pi_ = sqrt(2.0 * acos(0));
    
    // initialize statistics
    num_exact_computations_ = 0;
    
    // for the base case
    near_field_results_ = near_field;
    
    shell_pairs_.Copy(shell_pairs);
    ShuffleShellPairList(shell_pairs_, old_from_new_index_[0]);
    
    density_ = density;
    
  }
  
  
};

#endif
