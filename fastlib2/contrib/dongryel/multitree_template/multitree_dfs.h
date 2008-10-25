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

  double TotalNumTuplesHelper_(int b, ArrayList<Tree *> &nodes) {
    
    Tree *bkn = nodes[b];
    double result;
    int n = nodes.size();
    
    // If this is the last node in the list, then the result is the
    // number of points contained in this node.
    if(b == n - 1) {
      result = (double) bkn->count();
    }
    else {
      int j;
      int conflict = 0;
      int simple_product = 1;
    
      result = (double) bkn->count();
      
      for(j = b + 1 ; j < n && !conflict; j++) {
	Tree *knj = nodes[j];
	
	if (bkn->begin() >= knj->end() - 1) {
	  conflict = 1;
	}
	else if(nodes[j - 1]->end() - 1 > knj->begin()) {
	  simple_product = 0;
	}
      }
      
      if(conflict) {
	result = 0.0;
      }
      else if(simple_product) {
	for(j = b + 1; j < n; j++) {
	  result *= nodes[j]->count();
	}
      }
      else {
	int jdiff = -1; 
	
	// Undefined... will eventually point to the lowest j > b such
	// that nodes[j] is different from bkn
	for(j = b + 1; jdiff < 0 && j < n; j++) {
	  Tree *knj = nodes[j];
	  if(bkn->begin() != knj->begin() ||
	     bkn->end() - 1 != knj->end() - 1) {
	    jdiff = j;
	  }
	}
	
	if(jdiff < 0) {
	  result = math::BinomialCoefficient(bkn->count(), n - b);
	}
	else {
	  Tree *dkn = nodes[jdiff];
	  
	  if(dkn->begin() >= bkn->end() - 1) {
	    result = math::BinomialCoefficient(bkn->count(), jdiff - b);
	    if(result > 0.0) {
	      result *= TotalNumTuplesHelper_(jdiff, nodes);
	    }
	  }
	  else if(first_node_indices_strictly_surround_second_node_indices_
		  (bkn, dkn)) {
	    result = RecursiveTotalNumTuplesHelper_(b, nodes, b);
	  }
	  else if(first_node_indices_strictly_surround_second_node_indices_
		  (dkn, bkn)) {
	    result = RecursiveTotalNumTuplesHelper_(b, nodes, jdiff);
	  }
	}
      }
    }
    return result;
  }

  double RecursiveTotalNumTuplesHelper_(int b, ArrayList<Tree *> &nodes, 
					int i) {
    
    double result = 0.0;
    Tree *kni = nodes[i];
    nodes[i] = kni->left();
    result += TotalNumTuplesHelper_(b, nodes);
    nodes[i] = kni->right();
    result += TotalNumTuplesHelper_(b, nodes);
    nodes[i] = kni;
    return result;
  }

  /** @brief Returns the total number of valid $n$-tuples, in which
   *         the indices are chosen in the depth-first order.
   */
  double TotalNumTuples(ArrayList<Tree *> &nodes) {

    return TotalNumTuplesHelper_(0, nodes);
  }
  
  void CopyNodeSet_(const ArrayList<Tree *> &source_list,
		    ArrayList<Tree *> *destination_list);

  void Heuristic_(const ArrayList<Tree *> &nodes, 
		  index_t *max_count_among_non_leaf, index_t *split_index);

  void MultiTreeDepthFirstBase_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results);

  void MultiTreeDepthFirstCanonical_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results);

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
    MultiTreeDepthFirstCanonical_(sets_, trees_, *query_results);

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
    globals_.Init();
  }

};

#include "multitree_dfs_impl.h"

#endif
