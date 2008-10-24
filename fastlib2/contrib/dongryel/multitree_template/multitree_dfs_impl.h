template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::Heuristic_
(const ArrayList<Tree *> &nodes, index_t *max_count_among_non_leaf, 
 index_t *split_index) {

  MultiTreeCommon::MultiTreeHelper<0, MultiTreeProblem::order>::HeuristicLoop
    (nodes, max_count_among_non_leaf, split_index);
}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::MultiTreeDepthFirstBase_
(const ArrayList<Matrix *> &sets, ArrayList<Tree *> &nodes,
 typename MultiTreeProblem::MultiTreeQueryResult &query_results) {

  MultiTreeCommon::MultiTreeHelper<0, MultiTreeProblem::order>::BaseLoop
    (globals_, sets, nodes, query_results);
}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::MultiTreeDepthFirstCanonical_
(const ArrayList<Matrix *> &sets, ArrayList<Tree *> &nodes,
 typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
  
  if(MultiTreeProblem::ConsiderTupleExact(globals_, nodes)) {
    return;
  }

  // Figure out which ones are non-leaves.
  index_t split_index = -1;
  index_t max_count_among_non_leaf = 0;
  Heuristic_(nodes, &max_count_among_non_leaf, &split_index);
  
  // All leaves, then base case.
  if(split_index < 0) {
    MultiTreeDepthFirstBase_(sets, nodes, query_results);
    return;
  }
  
  // Else, split an internal node and recurse.
  else {
    double new_num_tuples;
    
    // Copy to new nodes list before recursing.
    ArrayList<Tree *> new_nodes;
    MultiTreeCommon::CopyNodeSet<MultiTreeProblem::order, Tree>(nodes,
								&new_nodes);
    
    // Push down approximations downward for the node that is to be
    // expanded.
    nodes[split_index]->left()->stat().postponed.ApplyPostponed
      (nodes[split_index]->stat().postponed);
    nodes[split_index]->right()->stat().postponed.ApplyPostponed
      (nodes[split_index]->stat().postponed);
    nodes[split_index]->stat().postponed.SetZero();

    // Recurse to the left.
    new_nodes[split_index] = nodes[split_index]->left();
    new_num_tuples = MultiTreeCommon::TotalNumTuples(new_nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MultiTreeDepthFirstCanonical_(sets, new_nodes, query_results);
    }
    
    // Recurse to the right.
    new_nodes[split_index] = nodes[split_index]->right();
    new_num_tuples = MultiTreeCommon::TotalNumTuples(new_nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MultiTreeDepthFirstCanonical_(sets, new_nodes, query_results);
    }
    
    // Apply the postponed changes for both child nodes.
    typename MultiTreeProblem::MultiTreeQuerySummary tmp_left_child_summary
      (nodes[split_index]->left()->stat().summary);
    tmp_left_child_summary.ApplyPostponed
      (nodes[split_index]->left()->stat().postponed);
    typename MultiTreeProblem::MultiTreeQuerySummary tmp_right_child_summary
      (nodes[split_index]->right()->stat().summary);
    tmp_right_child_summary.ApplyPostponed
      (nodes[split_index]->right()->stat().postponed);

    // Refine statistics after recursing.
    nodes[split_index]->stat().summary.StartReaccumulate();
    nodes[split_index]->stat().summary.Accumulate(tmp_left_child_summary);
    nodes[split_index]->stat().summary.Accumulate(tmp_right_child_summary);
    
    return;
  }
}
