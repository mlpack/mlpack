#ifndef MULTITREE_DFS_H
#define MULTITREE_DFS_H

#include "fastlib/fastlib.h"
#include "contrib/dongryel/proximity_project/general_spacetree.h"
#include "contrib/dongryel/proximity_project/gen_kdtree.h"
#include "mlpack/allknn/allknn.h"
#include "contrib/nvasil/allkfn/allkfn.h"

template<typename MultiTreeProblem>
class MultiTreeDepthFirst {
  
 private:
 
  typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix, typename MultiTreeProblem::MultiTreeStat > Tree;

  ArrayList<Matrix *> targets_;

  ArrayList<Matrix *> sets_;

  ArrayList<Tree *> trees_;

  typename MultiTreeProblem::MultiTreeGlobal globals_;

  Vector total_n_minus_one_tuples_;

  double total_n_minus_one_tuples_root_;

  template<int start, int end>
  class MultiTreeHelper_ {
   public:
    static void NestedLoop(typename MultiTreeProblem::MultiTreeGlobal &globals,
			   const ArrayList<Matrix *> &sets, 
			   ArrayList<Tree *> &nodes,
			   typename MultiTreeProblem::MultiTreeQueryResult
			   &query_results) {
      
      index_t starting_point_index = 0;
      index_t ending_point_index = nodes[start]->end();
      if(start == 0) {
	starting_point_index = nodes[start]->begin();
      }
      else {
	if(nodes[start - 1] == nodes[start]) {
	  starting_point_index = globals.chosen_indices[start - 1] + 1;
	}
	else {
	  starting_point_index = nodes[start]->begin();
	}
      }

      for(index_t i = starting_point_index; i < ending_point_index; i++) {
	globals.chosen_indices[start] = i;
	MultiTreeHelper_<start + 1, end>::NestedLoop(globals, sets, nodes,
						     query_results);
      }
    }

    static void RecursionLoop(const ArrayList<Matrix *> &sets,
			      ArrayList<Tree *> &nodes,
			      double total_num_tuples,
			      const bool contains_non_leaf_in_the_list,
			      typename MultiTreeProblem::MultiTreeQueryResult
			      &query_results,
			      MultiTreeDepthFirst *algorithm_object) {
      
      // In case of a leaf, then nothing else to do here.
      if(nodes[start]->is_leaf()) {
	
	// Check whether there is a conflict with the node chosen before...
	if(start == 0 || 
	   !(nodes[start]->end() <= nodes[start - 1]->begin())) {
	  MultiTreeHelper_<start + 1, end>::RecursionLoop
	    (sets, nodes, total_num_tuples, contains_non_leaf_in_the_list,
	     query_results, algorithm_object);
	}
      }
      else {
       
	// Save the node before choosing its child.
	Tree *saved_node = nodes[start];

	// Push down the postponed information before recursing...
	nodes[start]->left()->stat().postponed.ApplyPostponed
	  (nodes[start]->stat().postponed);
	nodes[start]->right()->stat().postponed.ApplyPostponed
	  (nodes[start]->stat().postponed);
	nodes[start]->stat().postponed.SetZero();

	// Visit flags to whether visit the left and the right.
	bool visit_left = true;
	bool visit_right = true;
	double squared_distance_to_left = DBL_MAX;
	double squared_distance_to_right = DBL_MAX;
	
	if(start == 0 || 
	   !(saved_node->left()->end() <= nodes[start - 1]->begin())) {
	  visit_left = true;
	  if(start > 0) {
	    squared_distance_to_left = nodes[start - 1]->bound().
	      MinDistanceSq(saved_node->left()->bound());
	  }
	}

	if(start == 0 || 
	   !(saved_node->right()->end() <= nodes[start - 1]->begin())) {
	  visit_right = true;
	  if(start > 0) {
	    squared_distance_to_right = nodes[start - 1]->bound().
	      MinDistanceSq(saved_node->right()->bound());
	  }
	}
	
	if(visit_left || start == 0) {
	  if(visit_right || start == 0) {
	    if(squared_distance_to_left <= squared_distance_to_right) {
	      nodes[start] = saved_node->left();
	      MultiTreeHelper_<start + 1, end>::RecursionLoop
		(sets, nodes, total_num_tuples, true, query_results,
		 algorithm_object);
	      
	      nodes[start] = saved_node->right();
	      MultiTreeHelper_<start + 1, end>::RecursionLoop
		(sets, nodes, total_num_tuples, true, query_results,
		 algorithm_object);
	    }
	    else {
	      nodes[start] = saved_node->right();
	      MultiTreeHelper_<start + 1, end>::RecursionLoop
		(sets, nodes, total_num_tuples, true, query_results,
		 algorithm_object);
	      
	      nodes[start] = saved_node->left();
	      MultiTreeHelper_<start + 1, end>::RecursionLoop
		(sets, nodes, total_num_tuples, true, query_results,
		 algorithm_object);	      
	    }
	  }
	  else {

	    // Just visit the left branch.
	    nodes[start] = saved_node->left();
	    MultiTreeHelper_<start + 1, end>::RecursionLoop
	      (sets, nodes, total_num_tuples, true, query_results,
	       algorithm_object);
	  }
	}
	else {
	  if(visit_right) {
	    nodes[start] = saved_node->right();
	    MultiTreeHelper_<start + 1, end>::RecursionLoop
	      (sets, nodes, total_num_tuples, true, query_results,
	       algorithm_object);	    
	  }
	}
	
	// Put back the node in the list after recursing...
	nodes[start] = saved_node;

	// Apply the postponed changes for both child nodes.
	typename MultiTreeProblem::MultiTreeQuerySummary tmp_left_child_summary
	  (nodes[start]->left()->stat().summary);
	tmp_left_child_summary.ApplyPostponed
	  (nodes[start]->left()->stat().postponed);
	typename MultiTreeProblem::MultiTreeQuerySummary
	  tmp_right_child_summary(nodes[start]->right()->stat().summary);
	tmp_right_child_summary.ApplyPostponed
	  (nodes[start]->right()->stat().postponed);
	
	// Refine statistics after recursing.
	nodes[start]->stat().summary.StartReaccumulate();
	nodes[start]->stat().summary.Accumulate(tmp_left_child_summary);
	nodes[start]->stat().summary.Accumulate(tmp_right_child_summary);
      }
    }

  };

  template<int end>
  class MultiTreeHelper_<end, end> {
   public:
    static void NestedLoop(typename MultiTreeProblem::MultiTreeGlobal &globals,
			   const ArrayList<Matrix *> &sets, 
			   ArrayList<Tree *> &nodes,			   
			   typename MultiTreeProblem::MultiTreeQueryResult
			   &query_results) {
      
      // Exhaustively compute the contribution due to the selected
      // tuple.
      globals.kernel_aux.EvaluateMain(globals, sets, query_results);
    }

    static void RecursionLoop(const ArrayList<Matrix *> &sets,
			      ArrayList<Tree *> &nodes,
			      double total_num_tuples,
			      const bool contains_non_leaf_in_the_list,
			      typename MultiTreeProblem::MultiTreeQueryResult
			      &query_results,
			      MultiTreeDepthFirst *algorithm_object) {

      if(contains_non_leaf_in_the_list) {
	double new_total_num_tuples = algorithm_object->TotalNumTuples(nodes);
	if(new_total_num_tuples > 0) {
	  algorithm_object->MultiTreeDepthFirstCanonical_
	    (sets, nodes, query_results, new_total_num_tuples);
	}
      }
      else {
	algorithm_object->MultiTreeDepthFirstBase_(sets, nodes, query_results,
						   total_num_tuples);
      }      
    }
  };

  int first_node_indices_strictly_surround_second_node_indices_
  (Tree *first_node, Tree *second_node) {
    
    return (first_node->begin() < second_node->begin() && 
	    first_node->end() >= second_node->end()) ||
      (first_node->begin() <= second_node->begin() && 
       first_node->end() > second_node->end());
  }

  double LeaveOneOutTuplesBase_(const ArrayList<Tree *> &nodes);

  double RecursiveLeaveOneOutTuples_(ArrayList<Tree *> &nodes,
				     int examine_start_index);

  /** @brief Returns the total number of valid $n$-tuples, in which
   *         the indices are chosen in the depth-first order.
   */
  double TotalNumTuples(ArrayList<Tree *> &nodes) {
    total_n_minus_one_tuples_.SetZero();
    return RecursiveLeaveOneOutTuples_(nodes, 0);
  }
  
  void CopyNodeSet_(const ArrayList<Tree *> &source_list,
		    ArrayList<Tree *> *destination_list);

  void Heuristic_(Tree *nd, Tree *nd1, Tree *nd2, Tree **partner1,
		  Tree **partner2);

  void MultiTreeDepthFirstBase_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results,
   double total_num_tuples);

  void MultiTreeDepthFirstCanonical_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results,
   double total_num_tuples);

  void PreProcessTree_(Tree *node, const ArrayList<double> &squared_nn_dists,
		       const ArrayList<double> &squared_fn_dists);
  
  void PostProcessTree_
  (Tree *node, typename MultiTreeProblem::MultiTreeQueryResult &query_results);

 public:

  MultiTreeDepthFirst() {
  }
  
  ~MultiTreeDepthFirst() {

    // This must be fixed...
    delete trees_[0];
  }

  void Compute(const ArrayList<Matrix *> *query_sets,
	       typename MultiTreeProblem::MultiTreeQueryResult
	       *query_results) {
    
    // Build the query tree.
    if(query_sets != NULL) {
      index_t previous_size = sets_.size();
      sets_.Resize(previous_size + query_sets->size());
      trees_.Resize(previous_size + query_sets->size());

      for(index_t i = 0; i < query_sets->size(); i++) {
	sets_[previous_size + i] = (*query_sets)[i];
	trees_[previous_size + i] = proximity::MakeGenKdTree<double, Tree, 
	  proximity::GenKdTreeMedianSplitter>(*(sets_[sets_.size() - 1]), 10,
					      NULL, NULL);
      }
    }

    // Assume that the query is the 0-th index.
    query_results->Init((sets_[sets_.size() - 1])->n_cols());

    // Preprocess the query trees.
    

    // Call the canonical algorithm.
    double total_num_tuples = TotalNumTuples(trees_);
    total_n_minus_one_tuples_root_ = total_n_minus_one_tuples_[0];

    printf("There are %g tuples...\n",
	   math::BinomialCoefficient((sets_[0])->n_cols() - 1, 
				     MultiTreeProblem::order));
    MultiTreeDepthFirstCanonical_(sets_, trees_, *query_results,
				  total_num_tuples);

    // Postprocess the query trees, also postprocessing the final
    // query results.
    PostProcessTree_(trees_[0], *query_results);
  }

  void NaiveCompute(typename MultiTreeProblem::MultiTreeQueryResult
		    *query_results) {
    
    // Assume that the query is the last index.
    query_results->Init((sets_[sets_.size() - 1])->n_cols());

    // Preprocess the query trees.    

    // Call the canonical algorithm.
    double total_num_tuples = TotalNumTuples(trees_);
    total_n_minus_one_tuples_root_ = total_n_minus_one_tuples_[0];

    printf("There are %g tuples...\n",
	   math::BinomialCoefficient((sets_[0])->n_cols() - 1, 
				     MultiTreeProblem::order));
    MultiTreeDepthFirstBase_(sets_, trees_, *query_results, total_num_tuples);

    // Postprocess the query trees, also postprocessing the final
    // query results.
    PostProcessTree_(trees_[0], *query_results);
  }

  void Init(const ArrayList<Matrix *> &sets,
	    const ArrayList<Matrix *> *targets) {
    
    // Copy the dataset and build the trees.
    sets_.Init(MultiTreeProblem::order);
    targets_.Init(MultiTreeProblem::order);
    trees_.Init(MultiTreeProblem::order);
    for(index_t i = 0; i < MultiTreeProblem::order; i++) {
      sets_[i] = sets[i];
      if(targets != NULL) {
	targets_[i] = (*targets)[i];
      }
    }

    // This could potentially be improved by checking which matrices
    // are the same...
    trees_[0] = proximity::MakeGenKdTree<double, Tree, 
      proximity::GenKdTreeMedianSplitter>(*(sets_[0]), 10, NULL, NULL);
    for(index_t i = 1; i < MultiTreeProblem::order; i++) {
      trees_[i] = trees_[0];
    }
    
    // Initialize the global parameters.
    globals_.Init((sets_[0])->n_cols());

    // Initialize the total number of (n - 1) tuples for each node
    // index.
    total_n_minus_one_tuples_.Init(sets.size());
  }

};

#include "multitree_dfs_impl.h"

#endif
