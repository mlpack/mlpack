#ifndef MULTITREE_DFS_H
#define MULTITREE_DFS_H

#include "fastlib/fastlib.h"
#include "contrib/dongryel/proximity_project/general_spacetree.h"
#include "contrib/dongryel/proximity_project/gen_kdtree.h"

template<typename MultiTreeProblem>
class MultiTreeDepthFirst {
  
 private:
 
  typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix, typename MultiTreeProblem::MultiTreeStat > Tree;

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

  void Heuristic_(const ArrayList<Tree *> &nodes, 
		  index_t *max_count_among_non_leaf, index_t *split_index);

  void MultiTreeDepthFirstBase_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results,
   double total_num_tuples);

  void MultiTreeDepthFirstCanonical_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results,
   double total_num_tuples);

  void PreProcessTree_(Tree *node);
  
  void PostProcessTree_
  (Tree *node, typename MultiTreeProblem::MultiTreeQueryResult &query_results);

 public:

  MultiTreeDepthFirst() {
  }
  
  ~MultiTreeDepthFirst() {

    // This must be fixed...
    delete trees_[0];
  }

  void Compute(typename MultiTreeProblem::MultiTreeQueryResult
	       *query_results) {
    
    // Assume that the query is the 0-th index.
    query_results->Init((sets_[0])->n_cols());

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

  void Init(const ArrayList<Matrix *> &sets) {
    
    // Copy the dataset and build the trees.
    sets_.Init(MultiTreeProblem::order);
    trees_.Init(MultiTreeProblem::order);
    for(index_t i = 0; i < MultiTreeProblem::order; i++) {
      sets_[i] = sets[i];
    }

    // This could potentially be improved by checking which matrices
    // are the same...
    trees_[0] = proximity::MakeGenKdTree<double, Tree, 
      proximity::GenKdTreeMidpointSplitter>(*(sets_[0]), 20, NULL, NULL);
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
