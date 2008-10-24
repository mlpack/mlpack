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

  void Heuristic_(const ArrayList<Tree *> &nodes, 
		  index_t *max_count_among_non_leaf, index_t *split_index);

  void MultiTreeDepthFirstBase_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results);

  void MultiTreeDepthFirstCanonical_
  (const ArrayList<Matrix *> &sets, ArrayList<Tree *> &trees,
   typename MultiTreeProblem::MultiTreeQueryResult &query_results);

 public:

  void Compute(typename MultiTreeProblem::MultiTreeQueryResult
	       *query_results) {
    
    // Assume that the query is the 0-th index.
    query_results->Init((sets_[0])->n_cols());

    // Preprocess the query trees.
    

    // Call the canonical algorithm.
    MultiTreeDepthFirstCanonical_(sets_, trees_, *query_results);

    // Postprocess the query trees, also postprocessing the final
    // query results.
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
  }

};

#include "multitree_dfs_impl.h"

#endif
