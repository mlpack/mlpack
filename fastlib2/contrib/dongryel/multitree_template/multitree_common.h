#ifndef MULTI_TREE_COMMON_H
#define MULTI_TREE_COMMON_H

#include "fastlib/fastlib.h"

#include "upper_triangular_square_matrix.h"

class MultiTreeCommon {

 private:

  template<int start, int end>
  class MultiTreeHelper {
    
   public:
    template<typename MultiTreeGlobal, typename MultiTreeQueryResult, 
	     typename Tree>
    static void BaseLoop(MultiTreeGlobal &globals,
			 const ArrayList<Matrix *> &sets, 
			 ArrayList<Tree *> &nodes,
			 MultiTreeQueryResult &query_results) {
      
      for(index_t i = nodes[start]->begin(); i < nodes[start]->end(); i++) {
	globals.chosen_indices[start] = i;
	MultiTreeHelper<start + 1, end>::BaseLoop(globals, sets, nodes,
						  query_results);
      }      
    }

    template<typename Tree>
    static void HeuristicLoop(const ArrayList<Tree *> &nodes,
			      index_t *max_count_among_non_leaf,
			      index_t *split_index) {
      
      if(!(nodes[start]->is_leaf()) &&
	 nodes[start]->count() > *max_count_among_non_leaf) {
	*max_count_among_non_leaf = nodes[start]->count();
	*split_index = start;
      }
      MultiTreeHelper<start + 1, end>::HeuristicLoop(nodes, 
						     max_count_among_non_leaf,
						     split_index);
    }

    template<typename Tree>
    static void CopyNodeSetLoop(const ArrayList<Tree *> &source_list,
				ArrayList<Tree *> *destination_list) {
      
      (*destination_list)[start] = source_list[start];
      MultiTreeHelper<start + 1, end>::CopyNodeSetLoop(source_list, 
						       destination_list);
    }

  };
  
  template<int end>
  class MultiTreeHelper<end, end> {
    
   public:
    
    template<typename MultiTreeGlobal, typename MultiTreeQueryResult, 
	     typename Tree>
    static void BaseLoop(MultiTreeGlobal &globals,
			 const ArrayList<Matrix *> &sets, 
			 ArrayList<Tree *> &nodes,
			 MultiTreeQueryResult &query_results) {

      // Exhaustively compute the contribution due to the selected
      // tuple.
      globals.kernel_aux.Evaluate(sets, globals);
    }

    template<typename Tree>
    static void HeuristicLoop(const ArrayList<Tree *> &nodes,
			      index_t *max_count_among_non_leaf,
			      index_t *split_index) {
      
      // Nothing to do: return...
    }

    template<typename Tree>
    static void CopyNodeSetLoop(const ArrayList<Tree *> &source_list,
				ArrayList<Tree *> *destination_list) {

      // Nothing to do: return...
    }
    
  };

  template<int first_index, int second_index, int end>
  class MultiTreeHelperThreeIndex {
 
   public:
    static void PairwiseEvaluateLoop(const ArrayList<Matrix *> &sets,
				     const ArrayList<index_t> &indices,
				     UpperTriangularSquareMatrix 
				     &squared_distances) {
      
      // Evaluate the squared distance between (first_index, start).
      const Matrix *first_set = sets[first_index];
      const Matrix *second_set = sets[second_index];
      double squared_distance =
	la::DistanceSqEuclidean
	(first_set->n_rows(), 
	 first_set->GetColumnPtr(indices[first_index]),
	 second_set->GetColumnPtr(indices[second_index]));
      squared_distances.set(first_index, second_index, squared_distance);

      printf("Evaluated between %d and %d\n", first_index, second_index);
      
      MultiTreeHelperThreeIndex<first_index, second_index + 1, end>::
	PairwiseEvaluateLoop(sets, indices, squared_distances);
    }
  };
  
  template<int end>
  class MultiTreeHelperThreeIndex<end, end, end> {
   public:
    static void PairwiseEvaluateLoop(const ArrayList<Matrix *> &sets,
				     const ArrayList<index_t> &indices,
				     UpperTriangularSquareMatrix
				     &squared_distances) {
      
      // Do nothing...
    }
  };

  template<int first_index, int end>
  class MultiTreeHelperThreeIndex<first_index, first_index, end> {

   public:
    static void PairwiseEvaluateLoop(const ArrayList<Matrix *> &sets,
				     const ArrayList<index_t> &indices,
				     UpperTriangularSquareMatrix
				     &squared_distances) {

      // Do nothing, but just advance the second index by one...
      MultiTreeHelperThreeIndex<first_index, first_index + 1,
	end>::PairwiseEvaluateLoop(sets, indices, squared_distances);
    }
  };

  template<int first_index, int second_index>
  class MultiTreeHelperThreeIndex<first_index, second_index, second_index> {

   public:
    static void PairwiseEvaluateLoop(const ArrayList<Matrix *> &sets,
				     const ArrayList<index_t> &indices,
				     UpperTriangularSquareMatrix
				     &squared_distances) {
	
      MultiTreeHelperThreeIndex<first_index + 1, first_index + 1,
	second_index>::PairwiseEvaluateLoop(sets, indices, squared_distances);
    }
  };

 public:

  template<int order, typename Tree>
  static void CopyNodeSet(const ArrayList<Tree *> &source_list,
			  ArrayList<Tree *> *destination_list) {

    // Allocate space...
    destination_list->Init(source_list.size());

    MultiTreeHelper<0, order>::CopyNodeSetLoop(source_list, destination_list);
  }

  template<typename Tree>
  static int first_node_indices_strictly_surround_second_node_indices_
  (Tree *first_node, Tree *second_node) {
    
    return (first_node->begin() < second_node->begin() && 
	    first_node->end() >= second_node->end()) ||
      (first_node->begin() <= second_node->begin() && 
       first_node->end() > second_node->end());
  }

  template<typename Tree>
  static double TotalNumTuplesHelper_(int b, ArrayList<Tree *> &nodes) {
    
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
	      result *= MultiTreeCommon::TotalNumTuplesHelper_(jdiff, nodes);
	    }
	  }
	  else if(first_node_indices_strictly_surround_second_node_indices_
		  (bkn, dkn)) {
	    result = MultiTreeCommon::RecursiveTotalNumTuplesHelper_(b, nodes,
								     b);
	  }
	  else if(first_node_indices_strictly_surround_second_node_indices_
		  (dkn, bkn)) {
	    result = MultiTreeCommon::RecursiveTotalNumTuplesHelper_(b, nodes,
								     jdiff);
	  }
	}
      }
    }
    return result;
  }

  template<typename Tree>
  static double RecursiveTotalNumTuplesHelper_(int b, 
					       ArrayList<Tree *> &nodes, 
					       int i) {
    
    double result = 0.0;
    Tree *kni = nodes[i];
    nodes[i] = kni->left();
    result += MultiTreeCommon::TotalNumTuplesHelper_(b, nodes);
    nodes[i] = kni->right();
    result += MultiTreeCommon::TotalNumTuplesHelper_(b, nodes);
    nodes[i] = kni;
    return result;
  }

  /** @brief Returns the total number of valid $n$-tuples, in which
   *         the indices are chosen in the depth-first order.
   */
  template<typename Tree>
  static double TotalNumTuples(ArrayList<Tree *> &nodes) {

    return MultiTreeCommon::TotalNumTuplesHelper_(0, nodes);
  }

};

#endif
