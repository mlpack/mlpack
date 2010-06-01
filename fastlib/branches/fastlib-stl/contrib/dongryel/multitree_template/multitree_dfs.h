/** @file multitree_dfs.h
 *
 *  The template auto-generator for depth first solver for general
 *  multitree problems.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MULTITREE_DFS_H
#define MULTITREE_DFS_H

#include "fastlib/fastlib.h"
#include "contrib/dongryel/proximity_project/general_spacetree.h"
#include "contrib/dongryel/proximity_project/gen_kdtree.h"
#include "contrib/dongryel/nested_summation_template/strata.h"
#include "multitree_utility.h"

template<typename MultiTreeProblem>
class MultiTreeDepthFirst {
  
 public:
 
  typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix, typename MultiTreeProblem::MultiTreeQueryStat > QueryTree;

  typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix, typename MultiTreeProblem::MultiTreeReferenceStat > ReferenceTree;
  
  typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix, typename MultiTreeProblem::MultiTreeQueryStat > HybridTree;

  ArrayList<Matrix *> targets_;

  ArrayList<Matrix *> sets_;

  ArrayList<ReferenceTree *> reference_trees_;

  ArrayList<index_t> old_from_new_references_;

  ArrayList<HybridTree *> hybrid_trees_;

  ArrayList<index_t> old_from_new_hybrids_;

  ArrayList<index_t> new_from_old_hybrids_;

  typename MultiTreeProblem::MultiTreeGlobal globals_;

  Vector total_n_minus_one_tuples_;

  double total_n_minus_one_tuples_root_;

 private:

  template<int start, int end>
  class MultiTreeHelper_ {
   public:
    static void HybridNodeNestedLoop
    (typename MultiTreeProblem::MultiTreeGlobal &globals,
     const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
      
      index_t starting_point_index = 0;
      index_t ending_point_index = hybrid_nodes[start]->end();
      if(start == 0) {
	starting_point_index = hybrid_nodes[start]->begin();
      }
      else {
	if(hybrid_nodes[start - 1] == hybrid_nodes[start]) {
	  starting_point_index = 
	    globals.hybrid_node_chosen_indices[start - 1] + 1;
	}
	else {
	  starting_point_index = hybrid_nodes[start]->begin();
	}
      }

      for(index_t i = starting_point_index; i < ending_point_index; i++) {
	globals.hybrid_node_chosen_indices[start] = i;
	MultiTreeHelper_<start + 1, end>::HybridNodeNestedLoop
	  (globals, query_sets, sets, targets, hybrid_nodes, query_nodes,
	   reference_nodes, query_results);
      }
    }
    
    static void QueryNodeNestedLoop
    (typename MultiTreeProblem::MultiTreeGlobal &globals,
     const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
      
      index_t starting_point_index = query_nodes[start]->begin();
      index_t ending_point_index = query_nodes[start]->end();

      for(index_t i = starting_point_index; i < ending_point_index; i++) {
	globals.query_node_chosen_indices[start] = i;
	MultiTreeHelper_<start + 1, end>::QueryNodeNestedLoop
	  (globals, query_sets, sets, targets, hybrid_nodes, query_nodes,
	   reference_nodes, query_results);
      }
    }

    static void ReferenceNodeNestedLoop
    (typename MultiTreeProblem::MultiTreeGlobal &globals,
     const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
      
      index_t starting_point_index = reference_nodes[start]->begin();
      index_t ending_point_index = reference_nodes[start]->end();

      for(index_t i = starting_point_index; i < ending_point_index; i++) {
	globals.reference_node_chosen_indices[start] = i;
	MultiTreeHelper_<start + 1, end>::ReferenceNodeNestedLoop
	  (globals, query_sets, sets, targets, hybrid_nodes, query_nodes,
	   reference_nodes, query_results);
      }
    }

    static void HybridNodeRecursionLoop
    (const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     double total_num_tuples, const bool contains_non_leaf_in_the_list,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results,
     MultiTreeDepthFirst *algorithm_object) {
      
      // In case of a leaf, then nothing else to do here.
      if(hybrid_nodes[start]->is_leaf()) {
	
	// Check whether there is a conflict with the node chosen before...
	if(start == 0 || 
	   !(hybrid_nodes[start]->end() <= hybrid_nodes[start - 1]->begin())) {
	  MultiTreeHelper_<start + 1, end>::HybridNodeRecursionLoop
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, total_num_tuples, contains_non_leaf_in_the_list,
	     query_results, algorithm_object);
	}
      }
      else {
       
	// Save the node before choosing its child.
	HybridTree *saved_node = hybrid_nodes[start];

	// Push down the postponed information before recursing...
	hybrid_nodes[start]->left()->stat().postponed.ApplyPostponed
	  (hybrid_nodes[start]->stat().postponed);
	hybrid_nodes[start]->right()->stat().postponed.ApplyPostponed
	  (hybrid_nodes[start]->stat().postponed);
	hybrid_nodes[start]->stat().postponed.SetZero();

	// Visit flags to whether visit the left and the right.
	HybridTree *first_node = NULL;
	HybridTree *second_node = NULL;

	// Decide which branch to visit...
	HybridTree *previously_chosen_node = (start == 0) ?
	  NULL:hybrid_nodes[start - 1];
	algorithm_object->Heuristic_<true, HybridTree, HybridTree>
	  (previously_chosen_node, saved_node->left(), saved_node->right(),
	   &first_node, &second_node);
	
	// Visit the first node, if not null...
	if(first_node != NULL) {
	  hybrid_nodes[start] = first_node;
	  MultiTreeHelper_<start + 1, end>::HybridNodeRecursionLoop
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, total_num_tuples, true, query_results,
	     algorithm_object);
	}

	// Visit the other node, if not null...
	if(second_node != NULL) {
	  hybrid_nodes[start] = second_node;
	  MultiTreeHelper_<start + 1, end>::HybridNodeRecursionLoop
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, total_num_tuples, true, query_results,
	     algorithm_object);
	}
	
	// Put back the node in the list after recursing...
	hybrid_nodes[start] = saved_node;

	// Apply the postponed changes for both child nodes.
	typename MultiTreeProblem::MultiTreeQuerySummary tmp_left_child_summary
	  (hybrid_nodes[start]->left()->stat().summary);
	tmp_left_child_summary.ApplyPostponed
	  (hybrid_nodes[start]->left()->stat().postponed);
	typename MultiTreeProblem::MultiTreeQuerySummary
	  tmp_right_child_summary
	  (hybrid_nodes[start]->right()->stat().summary);
	tmp_right_child_summary.ApplyPostponed
	  (hybrid_nodes[start]->right()->stat().postponed);
	
	// Refine statistics after recursing.
	hybrid_nodes[start]->stat().summary.StartReaccumulate();
	hybrid_nodes[start]->stat().summary.Accumulate(tmp_left_child_summary);
	hybrid_nodes[start]->stat().summary.Accumulate
	  (tmp_right_child_summary);
      }
    }

    static void QueryNodeRecursionLoop
    (const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     double total_num_tuples, const bool contains_non_leaf_in_the_list,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results,
     MultiTreeDepthFirst *algorithm_object) {
      
      // In case of a leaf, then nothing else to do here.
      if(query_nodes[start]->is_leaf()) {
	
	MultiTreeHelper_<start + 1, end>::QueryNodeRecursionLoop
	  (query_sets, sets, targets, hybrid_nodes, query_nodes,
	   reference_nodes, total_num_tuples, contains_non_leaf_in_the_list,
	   query_results, algorithm_object);
      }
      else {
       
	// Save the node before choosing its child.
	QueryTree *saved_node = query_nodes[start];

	// Push down the postponed information before recursing...
	query_nodes[start]->left()->stat().postponed.ApplyPostponed
	  (query_nodes[start]->stat().postponed);
	query_nodes[start]->right()->stat().postponed.ApplyPostponed
	  (query_nodes[start]->stat().postponed);
	query_nodes[start]->stat().postponed.SetZero();

	// Visit flags to whether visit the left and the right.
	QueryTree *first_node = NULL;
	QueryTree *second_node = NULL;

	// Decide which branch to visit...
	HybridTree *previously_chosen_node = (start == 0) ?
	  NULL:query_nodes[start - 1];
	algorithm_object->Heuristic_<false, QueryTree, QueryTree>
	  (previously_chosen_node, saved_node->left(), saved_node->right(),
	   &first_node, &second_node);
	
	// Visit the first node, if not null...
	if(first_node != NULL) {
	  query_nodes[start] = first_node;
	  MultiTreeHelper_<start + 1, end>::QueryNodeRecursionLoop
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, total_num_tuples, true, query_results,
	     algorithm_object);
	}

	// Visit the other node, if not null...
	if(second_node != NULL) {
	  query_nodes[start] = second_node;
	  MultiTreeHelper_<start + 1, end>::QueryNodeRecursionLoop
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, total_num_tuples, true, query_results,
	     algorithm_object);
	}
	
	// Put back the node in the list after recursing...
	query_nodes[start] = saved_node;

	// Apply the postponed changes for both child nodes.
	typename MultiTreeProblem::MultiTreeQuerySummary tmp_left_child_summary
	  (query_nodes[start]->left()->stat().summary);
	tmp_left_child_summary.ApplyPostponed
	  (query_nodes[start]->left()->stat().postponed);
	typename MultiTreeProblem::MultiTreeQuerySummary
	  tmp_right_child_summary(query_nodes[start]->right()->stat().summary);
	tmp_right_child_summary.ApplyPostponed
	  (query_nodes[start]->right()->stat().postponed);
	
	// Refine statistics after recursing.
	query_nodes[start]->stat().summary.StartReaccumulate();
	query_nodes[start]->stat().summary.Accumulate(tmp_left_child_summary);
	query_nodes[start]->stat().summary.Accumulate(tmp_right_child_summary);
      }
    }

    static void ReferenceNodeRecursionLoop
    (const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     double total_num_tuples, const bool contains_non_leaf_in_the_list,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results,
     MultiTreeDepthFirst *algorithm_object) {
      
      // In case of a leaf, then nothing else to do here.
      if(reference_nodes[start]->is_leaf()) {
	
	MultiTreeHelper_<start + 1, end>::ReferenceNodeRecursionLoop
	  (query_sets, sets, targets, hybrid_nodes, query_nodes,
	   reference_nodes, total_num_tuples, contains_non_leaf_in_the_list,
	   query_results, algorithm_object);
      }
      else {
       
	// Save the node before choosing its child.
	ReferenceTree *saved_node = reference_nodes[start];

	// Visit flags to whether visit the left and the right.
	ReferenceTree *first_node = NULL;
	ReferenceTree *second_node = NULL;

	// Decide which branch to visit...
	QueryTree *previously_chosen_query_node = (start == 0) ?
	  query_nodes[MultiTreeProblem::num_query_sets - 1]:NULL;
	ReferenceTree *previously_chosen_reference_node = (start > 0) ?
	  reference_nodes[start - 1]:NULL;

	if(previously_chosen_query_node != NULL) {
	  algorithm_object->Heuristic_<false, QueryTree, ReferenceTree>
	    (previously_chosen_query_node, saved_node->left(),
	     saved_node->right(), &first_node, &second_node);
	}
	else {
	  algorithm_object->Heuristic_<false, ReferenceTree, ReferenceTree>
	    (previously_chosen_reference_node, saved_node->left(),
	     saved_node->right(), &first_node, &second_node);
	}
	
	// Visit the first node, if not null...
	if(first_node != NULL) {
	  reference_nodes[start] = first_node;
	  MultiTreeHelper_<start + 1, end>::ReferenceNodeRecursionLoop
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, total_num_tuples, true, query_results,
	     algorithm_object);
	}

	// Visit the other node, if not null...
	if(second_node != NULL) {
	  reference_nodes[start] = second_node;
	  MultiTreeHelper_<start + 1, end>::ReferenceNodeRecursionLoop
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, total_num_tuples, true, query_results,
	     algorithm_object);
	}
	
	// Put back the node in the list after recursing...
	reference_nodes[start] = saved_node;
      }
    }

  };

  template<int end>
  class MultiTreeHelper_<end, end> {
   public:
    static void HybridNodeNestedLoop
    (typename MultiTreeProblem::MultiTreeGlobal &globals,
     const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes, 
     ArrayList<ReferenceTree *> &reference_nodes, 
     typename MultiTreeProblem::MultiTreeQueryResult &query_results) {

      // Exhaustively compute the contribution due to the selected
      // tuple.
      MultiTreeProblem::HybridNodeEvaluateMain(globals, query_sets, sets,
					       targets, query_results);
    }

    static void QueryNodeNestedLoop
    (typename MultiTreeProblem::MultiTreeGlobal &globals,
     const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes, 
     ArrayList<ReferenceTree *> &reference_nodes, 
     typename MultiTreeProblem::MultiTreeQueryResult &query_results) {

      MultiTreeHelper_<0, MultiTreeProblem::num_reference_sets>::
	ReferenceNodeNestedLoop(globals, query_sets, sets, targets,
				hybrid_nodes, query_nodes, reference_nodes,
				query_results);
    }

    static void ReferenceNodeNestedLoop
    (typename MultiTreeProblem::MultiTreeGlobal &globals,
     const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes, 
     ArrayList<ReferenceTree *> &reference_nodes, 
     typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
      MultiTreeProblem::ReferenceNodeEvaluateMain(globals, query_sets, sets,
						  targets, query_results);
    }

    static void HybridNodeRecursionLoop
    (const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     double total_num_tuples, const bool contains_non_leaf_in_the_list,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results,
     MultiTreeDepthFirst *algorithm_object) {

      MultiTreeHelper_<0, MultiTreeProblem::num_query_sets>::
	QueryNodeRecursionLoop
	(query_sets, sets, targets, hybrid_nodes, query_nodes, reference_nodes,
	 total_num_tuples, contains_non_leaf_in_the_list, query_results,
	 algorithm_object);
    }

    static void QueryNodeRecursionLoop
    (const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     double total_num_tuples, const bool contains_non_leaf_in_the_list,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results,
     MultiTreeDepthFirst *algorithm_object) {
      
      MultiTreeHelper_<0, MultiTreeProblem::num_reference_sets>::
	ReferenceNodeRecursionLoop
	(query_sets, sets, targets, hybrid_nodes, query_nodes, reference_nodes,
	 total_num_tuples, contains_non_leaf_in_the_list, query_results,
	 algorithm_object);
    }
    
    static void ReferenceNodeRecursionLoop
    (const ArrayList<Matrix *> &query_sets,
     const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
     ArrayList<HybridTree *> &hybrid_nodes,
     ArrayList<QueryTree *> &query_nodes,
     ArrayList<ReferenceTree *> &reference_nodes,
     double total_num_tuples, const bool contains_non_leaf_in_the_list,
     typename MultiTreeProblem::MultiTreeQueryResult &query_results,
     MultiTreeDepthFirst *algorithm_object) {

      if(contains_non_leaf_in_the_list) {
		
	double new_total_num_tuples =
	  algorithm_object->TotalNumTuples(hybrid_nodes, query_nodes,
					   reference_nodes);

	if(new_total_num_tuples > 0) {
	  algorithm_object->MultiTreeDepthFirstCanonical_
	    (query_sets, sets, targets, hybrid_nodes, query_nodes,
	     reference_nodes, query_results, new_total_num_tuples);	  
	}
      }
      else {
	algorithm_object->MultiTreeDepthFirstBase_
	  (query_sets, sets, targets, hybrid_nodes, query_nodes,
	   reference_nodes, query_results, total_num_tuples);
      }      
    }
  };

  template<typename Tree>
  int first_node_indices_strictly_surround_second_node_indices_
  (Tree *first_node, Tree *second_node) {
    
    return (first_node->begin() < second_node->begin() && 
	    first_node->end() >= second_node->end()) ||
      (first_node->begin() <= second_node->begin() && 
       first_node->end() > second_node->end());
  }

  template<bool is_hybrid_node, typename TreeType1, typename TreeType2>
  void Heuristic_(TreeType1 *nd, TreeType2 *nd1, TreeType2 *nd2,
		  TreeType2 **partner1, TreeType2 **partner2);

  double LeaveOneOutTuplesBase_(const ArrayList<HybridTree *> &hybrid_nodes);

  double RecursiveLeaveOneOutTuples_(ArrayList<HybridTree *> &hybrid_nodes,
				     int examine_start_index);

  /** @brief Returns the total number of valid $n$-tuples, in which
   *         the indices are chosen in the depth-first order.
   */
  double TotalNumTuples(ArrayList<HybridTree *> &hybrid_nodes,
			ArrayList<QueryTree *> &query_nodes,
			ArrayList<ReferenceTree *> &reference_nodes) {
    total_n_minus_one_tuples_.SetZero();

    if(MultiTreeProblem::num_hybrid_sets > 0) {
      return RecursiveLeaveOneOutTuples_(hybrid_nodes, 0);
    }
    else {
      double product_reference_node_count = 1.0;
      for(index_t i = 0; i < reference_nodes.size(); i++) {
	product_reference_node_count *= reference_nodes[i]->count();
      }
      return product_reference_node_count;
    }
  }

  void MultiTreeDepthFirstBase_
  (const ArrayList<Matrix *> &query_sets,
   const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
   ArrayList<HybridTree *> &hybrid_trees,
   ArrayList<QueryTree *> &query_trees,
   ArrayList<ReferenceTree *> &reference_tress,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results,
   double total_num_tuples);

  void MultiTreeDepthFirstCanonical_
  (const ArrayList<Matrix *> &query_sets,
   const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
   ArrayList<HybridTree *> &hybrid_trees,
   ArrayList<QueryTree *> &query_trees,
   ArrayList<ReferenceTree *> &reference_trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results,
   double total_num_tuples);

  template<typename Tree>
  void PreProcessQueryTree_(Tree *node);

  template<typename Tree>
  void PreProcessQueryTreeMonochromatic_
  (Tree *node, const typename MultiTreeProblem::MultiTreeQueryResult 
   &results);

  template<typename Tree>
  void PreProcessReferenceTree_(Tree *node, index_t reference_tree_index);

  template<typename Tree>
  void PostProcessTree_
  (const Matrix &qset, Tree *node,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results);

 public:

  /** @brief The default constructor which does nothing.
   */
  MultiTreeDepthFirst() {
  }
  
  ~MultiTreeDepthFirst() {

    // This must be fixed...
    if(hybrid_trees_.size() > 0) {
      delete hybrid_trees_[0];
      delete sets_[0];
    }

    if(reference_trees_.size() > 0) {
      for(index_t i = 0; i < reference_trees_.size(); i++) {
	delete reference_trees_[i];
	delete sets_[i];
      }
    }
    
    if(targets_.size() > 0) {
      for(index_t i = 0; i < targets_.size(); i++) {
	delete targets_[i];
      }
    }
  }

  void Compute(const ArrayList<const Matrix *> *query_sets_in,
	       typename MultiTreeProblem::MultiTreeQueryResult
	       *query_results, bool monochromatic_trick = false) {

    ArrayList<Matrix *> query_sets;
    ArrayList<QueryTree *> query_trees;
    ArrayList<index_t> old_from_new_queries;
    ArrayList<index_t> new_from_old_queries;
    
    // Build the query tree.
    if(query_sets_in != NULL) {
      query_sets.Init(query_sets_in->size());
      query_trees.Init(query_sets_in->size());

      for(index_t i = 0; i < query_sets.size(); i++) {
	query_sets[i] = new Matrix();
	query_sets[i]->Copy(*((*query_sets_in)[i]));
	query_trees[i] = proximity::MakeGenKdTree<double, QueryTree,
	  proximity::GenKdTreeMedianSplitter>(*(query_sets[i]), 40,
					      &old_from_new_queries,
					      &new_from_old_queries);
      }
    }
    else {
      query_sets.Init();
      query_trees.Init();
    }

    // Assume that the query is the 0-th index.
    if(query_sets.size() > 0) {
      query_results->Init((query_sets[0])->n_cols());
    }
    else {
      query_results->Init((sets_[0])->n_cols());
    }

    // Preprocess the query trees. Note that the bounds will be
    // overwritten if the monochromatic tree is applied...
    if(query_sets_in != NULL) {
      PreProcessQueryTree_(query_trees[0]);
    }

    /*
    // If the monochromatic trick is required, then create the initial
    // strata and do a monochromatic on each strata to build up the
    // bounds.
    if(monochromatic_trick) {
      Strata frontier;
      frontier.Init(hybrid_trees_[0], MultiTreeProblem::order - 1, 0.1);

      printf("%d strata created for the monochromatic trick...\n",
	     frontier.total_num_stratum);

      // Save the root nodes before doing the trick...
      ArrayList<HybridTree *> saved_nodes;
      saved_nodes.Init(hybrid_trees_.size());
      for(index_t i = 0; i < hybrid_trees_.size(); i++) {
	saved_nodes[i] = hybrid_trees_[i];
      }

      for(index_t i = 0; i < frontier.total_num_stratum; i++) {
	
	for(index_t j = 0; j < hybrid_trees_.size(); j++) {
	  hybrid_trees_[j] = frontier.node_list[i];
	}

	double total_num_tuples = TotalNumTuples(hybrid_trees_, query_trees,
						 reference_trees_);
	total_n_minus_one_tuples_root_ = total_n_minus_one_tuples_[0];

	MultiTreeDepthFirstBase_(query_sets, sets_, targets_, hybrid_trees_,
				 query_trees, reference_trees_, *query_results,
				 total_num_tuples);
      }
      for(index_t i = 0; i < hybrid_trees_.size(); i++) {
        hybrid_trees_[i] = saved_nodes[i];
      }

      PreProcessQueryTreeMonochromatic_(hybrid_trees_[0], *query_results);
    }
    */

    // Call the canonical algorithm.
    double total_num_tuples = TotalNumTuples(hybrid_trees_, query_trees,
                                             reference_trees_);
    total_n_minus_one_tuples_root_ = total_n_minus_one_tuples_[0];

    printf("There are %g tuples...\n",
           math::BinomialCoefficient((sets_[0])->n_cols() - 1,
                                     MultiTreeProblem::order - 1));

    MultiTreeDepthFirstCanonical_(query_sets, sets_, targets_, hybrid_trees_,
				  query_trees, reference_trees_,
				  *query_results, total_num_tuples);

    // Postprocess the query trees, also postprocessing the final
    // query results and free memory.
    if(query_sets_in != NULL) {
      PostProcessTree_(*(query_sets[0]), query_trees[0], *query_results);
      for(index_t i = 0; i < query_sets.size(); i++) {
        delete query_trees[i];
        delete query_sets[i];
      }
    }
    else {
      PostProcessTree_(*(sets_[0]), hybrid_trees_[0], *query_results);
    }

    if(query_sets.size() > 0) {
      query_results->Finalize(globals_, old_from_new_queries);
    }
    else {
      query_results->Finalize(globals_, old_from_new_hybrids_);
    }
  }

  void NaiveCompute(const ArrayList<const Matrix *> *query_sets_in,
		    typename MultiTreeProblem::MultiTreeQueryResult
		    *query_results) {

    ArrayList<QueryTree *> query_trees;
    ArrayList<Matrix *> query_sets;
    ArrayList<index_t> old_from_new_queries;
    ArrayList<index_t> new_from_old_queries;

    // Build the query tree.
    if(query_sets_in != NULL) {
      query_sets.Init(query_sets_in->size());
      query_trees.Init(query_sets_in->size());

      for(index_t i = 0; i < query_sets_in->size(); i++) {
	query_sets[i] = new Matrix();
	query_sets[i]->Copy(*((*query_sets_in)[i]));
	query_trees[i] = proximity::MakeGenKdTree<double,
	  QueryTree, proximity::GenKdTreeMedianSplitter>
	  (*(query_sets[i]), ((query_sets[i])->n_cols()) * 2, 
	   &old_from_new_queries, &new_from_old_queries);
      }
    }
    else {
      query_sets.Init();
      query_trees.Init();
    }

    // Assume that the query is the 0-th index.
    if(query_sets.size() > 0) {
      query_results->Init((query_sets[0])->n_cols());
    }
    else {
      query_results->Init((sets_[0])->n_cols());
    }

    // Preprocess the query trees.
    if(query_sets_in != NULL) {
      PreProcessQueryTree_(query_trees[0]);
    }

    // Call the canonical algorithm.
    double total_num_tuples = TotalNumTuples(hybrid_trees_, query_trees,
					     reference_trees_);
    total_n_minus_one_tuples_root_ = total_n_minus_one_tuples_[0];

    printf("There are %g tuples...\n",
	   math::BinomialCoefficient((sets_[0])->n_cols() - 1, 
				     MultiTreeProblem::order - 1));
    MultiTreeDepthFirstBase_(query_sets, sets_, targets_, hybrid_trees_,
			     query_trees, reference_trees_, *query_results,
			     total_num_tuples);

    // Postprocess the query trees, also postprocessing the final
    // query results.
    if(query_sets_in != NULL) {
      PostProcessTree_(*(query_sets[0]), query_trees[0], *query_results);

      for(index_t i = 0; i < query_sets.size(); i++) {
	delete query_trees[i];
	delete query_sets[i];
      }
    }
    else {
      PostProcessTree_(*(sets_[0]), hybrid_trees_[0], *query_results);
    }

    if(query_sets.size() > 0) {
      query_results->Finalize(globals_, old_from_new_queries);
    }
    else {
      query_results->Finalize(globals_, old_from_new_hybrids_);
    }
  }

  void InitMultiChromatic(const ArrayList<const Matrix *> &sets,
			  const ArrayList<const Matrix *> *targets,
			  struct datanode *module_in) {

    // In this case, the hybrid set is empty...
    hybrid_trees_.Init();

    // Copy the dataset and build the trees.
    sets_.Init(MultiTreeProblem::num_reference_sets);
    targets_.Init(MultiTreeProblem::num_reference_sets);
    reference_trees_.Init(MultiTreeProblem::num_reference_sets);
    for(index_t i = 0; i < MultiTreeProblem::num_reference_sets; i++) {
      sets_[i] = new Matrix();
      sets_[i]->Copy((*sets[i]));
      if(targets != NULL) {
	targets_[i] = new Matrix();
	targets_[i]->Copy((*((*targets)[i])));
      }
    }

    // Initialize the global parameters.
    globals_.Init((sets_[0])->n_cols(), sets[0]->n_rows(), targets_,
		  module_in);

    // This could potentially be improved by checking which matrices
    // are the same...
    reference_trees_[0] = proximity::MakeGenKdTree<double, ReferenceTree,
      proximity::GenKdTreeMedianSplitter>(*(sets_[0]), 40, 
					  &old_from_new_references_, NULL);
    PreProcessReferenceTree_(reference_trees_[0], 0);
    for(index_t i = 1; i < MultiTreeProblem::num_reference_sets; i++) {
      reference_trees_[i] = reference_trees_[0];
    }

    // Shuffle the target values according to the permutation
    // shuffling of the reference points.
    if(targets != NULL) {
      MultiTreeUtility::ShuffleAccordingToPermutation
	(*(targets_[0]), old_from_new_references_);
    }
    
    // Initialize the total number of (n - 1) tuples for each node
    // index.
    total_n_minus_one_tuples_.Init(sets.size());
  }

  void InitMonoChromatic(const ArrayList<const Matrix *> &sets,
			 const ArrayList<const Matrix *> *targets,
			 struct datanode *module_in) {

    // In this case, the reference set is empty...
    reference_trees_.Init();

    // Copy the dataset and build the trees.
    sets_.Init(MultiTreeProblem::num_hybrid_sets);
    targets_.Init(MultiTreeProblem::num_hybrid_sets);
    hybrid_trees_.Init(MultiTreeProblem::num_hybrid_sets);
    for(index_t i = 0; i < MultiTreeProblem::num_hybrid_sets; i++) {
      if(i == 0) {
	sets_[i] = new Matrix();
	sets_[i]->Copy((*sets[i]));
      }
      else {
	sets_[i] = sets_[0];
      }
      if(targets != NULL) {
	if(i == 0) {
	  targets_[i] = new Matrix();
	  targets_[i]->Copy((*((*targets)[i])));
	}
	else {
	  targets_[i] = targets_[0];
	}	
      }
      else {
	targets_[i] = NULL;
      }
    }

    // This could potentially be improved by checking which matrices
    // are the same...
    hybrid_trees_[0] = proximity::MakeGenKdTree<double, HybridTree,
      proximity::GenKdTreeMedianSplitter>(*(sets_[0]), 20, 
					  &old_from_new_hybrids_,
					  &new_from_old_hybrids_);
    for(index_t i = 1; i < MultiTreeProblem::num_hybrid_sets; i++) {
      hybrid_trees_[i] = hybrid_trees_[0];
    }
    
    // Initialize the global parameters.
    globals_.Init((sets_[0])->n_cols(), (sets_[0])->n_rows(), targets_,
		  module_in);

    // Initialize the total number of (n - 1) tuples for each node
    // index.
    total_n_minus_one_tuples_.Init(sets.size());
  }
};

#include "multitree_dfs_impl.h"

#endif
